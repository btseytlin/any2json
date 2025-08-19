try:
    import unsloth
except ImportError:
    pass

import os
from dataclasses import dataclass

import click
from dotenv import load_dotenv
import wandb
from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.hf_argparser import HfArgumentParser
import torch
from any2json.utils import configure_loggers, logger
from any2json.training.augment import build_augmentor, apply_augmentations
from any2json.training.utils import (
    augment_train_split,
    filter_tokenized_splits_by_length,
    load_hf_dataset,
    apply_debug_limit,
    make_group_split,
    EvalLoggerCallback,
    DebugTokensCallback,
    estimate_token_lengths,
    build_tokenize_fn,
    CausalLMDataCollator,
)


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


@dataclass
class PipelineConfig:
    dataset_path: str
    model_name: str
    max_source_length: int | None
    max_target_length: int | None
    drop_schema_proba: float
    schema_missing_token: str
    input_aug: list[str]
    output_aug: list[str]
    debug_limit: int | None
    val_size: int
    wandb_project: str
    pad_to_multiple_of: int
    debug_tokens: bool
    unsloth: bool
    hf_args: TrainingArguments


def validate_pipeline_config(cfg: PipelineConfig) -> None:
    if not 0 <= cfg.drop_schema_proba <= 1:
        raise ValueError("drop_schema_proba must be in [0, 1]")
    if cfg.debug_limit is not None and cfg.debug_limit < 0:
        raise ValueError("debug_limit must be >= 0")


def validate_training_args(args: TrainingArguments) -> None:
    if getattr(args, "fp16", False) and getattr(args, "bf16", False):
        raise ValueError("Cannot enable both fp16 and bf16")
    if getattr(args, "use_cpu", False) and (
        getattr(args, "fp16", False) or getattr(args, "bf16", False)
    ):
        raise ValueError("Mixed precision requires GPU; disable use_cpu or fp16/bf16")
    if getattr(args, "bf16", False) and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requested but not supported on this hardware")
    if getattr(args, "fp16", False) and not torch.cuda.is_available():
        raise ValueError("fp16 requested but CUDA is not available")


def prepare_splits(ds: DatasetDict, seed: int, test_size: int = 5000) -> DatasetDict:
    base = DatasetDict({"train": ds["train"]}) if "train" in ds else ds
    size = len(base["train"]) if "train" in base else 0
    test_size = min(size, test_size) if size > test_size else max(1, size // 20)
    return make_group_split(base, test_size=test_size, seed=seed)


def tokenize_splits(
    ds: DatasetDict,
    tokenizer: AutoTokenizer,
    cfg: PipelineConfig,
    num_proc: int = 8,
) -> DatasetDict:
    fn = build_tokenize_fn(tokenizer, debug=cfg.debug_tokens)
    train_tok = ds["train"].map(
        fn,
        batched=True,
        remove_columns=ds["train"].column_names,
        num_proc=num_proc,
    )
    val_tok = ds["validation"].map(
        fn,
        batched=True,
        remove_columns=ds["validation"].column_names,
        num_proc=num_proc,
    )
    return DatasetDict({"train": train_tok, "validation": val_tok})


def create_trainer(
    ds: DatasetDict,
    tokenized: DatasetDict,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    args: TrainingArguments,
    pad_to_multiple_of: int = 8,
    debug_tokens: bool = False,
):
    collator = CausalLMDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.add_callback(
        EvalLoggerCallback(
            tokenizer, ds["validation"], pad_to_multiple_of=pad_to_multiple_of
        )
    )
    if debug_tokens:
        trainer.add_callback(DebugTokensCallback(tokenizer))
    return trainer


def prepare_dataset(
    pcfg: PipelineConfig,
    args: TrainingArguments,
    tokenizer: AutoTokenizer,
) -> DatasetDict:
    raw = load_hf_dataset(pcfg.dataset_path)
    logger.info(f"Loaded {len(raw['train'])} train samples")

    if pcfg.debug_limit:
        raw = apply_debug_limit(raw, pcfg.debug_limit)
        logger.info(
            f"Applied debug limit: {pcfg.debug_limit}, now {len(raw['train'])} train samples"
        )
    logger.info(f"Preparing splits with val size: {pcfg.val_size}")
    ds = prepare_splits(raw, seed=args.seed, test_size=pcfg.val_size)
    logger.info(f"Prepared splits with val size: {pcfg.val_size}: {ds}")
    ds = augment_train_split(ds, pcfg, seed=args.seed)
    logger.info(f"Augmented train split: {ds}")

    logger.info(f"Tokenizing splits")
    tokenized = tokenize_splits(ds, tokenizer, pcfg)

    logger.info(
        f"Filtering tokenized data by length, max_source_length: {pcfg.max_source_length}, max_target_length: {pcfg.max_target_length}"
    )
    tokenized = filter_tokenized_splits_by_length(
        tokenized, pcfg.max_source_length, pcfg.max_target_length
    )
    logger.info(f"Filtered tokenized datasets: {tokenized}")
    return ds, tokenized


def prepare_model_and_tokenizer(
    pcfg: PipelineConfig, args: TrainingArguments
) -> tuple[AutoModelForCausalLM, AutoTokenizer | None]:
    if pcfg.unsloth:
        from unsloth import FastModel

        assert "unsloth" in pcfg.model_name, "Must use an unsloth model with --unsloth"

        max_seq_length = (
            pcfg.max_source_length + pcfg.max_target_length
            if pcfg.max_source_length and pcfg.max_target_length
            else None
        )

        model, tokenizer = FastModel.from_pretrained(
            pcfg.model_name,
            full_finetuning=True,
            use_gradient_checkpointing="unsloth",
            max_seq_length=max_seq_length,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(pcfg.model_name)
        model.config.use_cache = False
        if getattr(args, "gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        tokenizer = AutoTokenizer.from_pretrained(pcfg.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return model, tokenizer


def run_training(pcfg: PipelineConfig, args: TrainingArguments) -> None:
    os.environ.setdefault("WANDB_PROJECT", pcfg.wandb_project)
    os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "False")

    logger.info(f"Pipeline config: {pcfg}")

    logger.info(f"Loading model and tokenizer")
    model, tokenizer = prepare_model_and_tokenizer(pcfg, args)

    pcfg.max_source_length = pcfg.max_source_length or tokenizer.model_max_length // 2
    pcfg.max_source_length = min(pcfg.max_source_length, tokenizer.model_max_length)
    pcfg.max_target_length = pcfg.max_target_length or tokenizer.model_max_length // 2
    pcfg.max_target_length = min(pcfg.max_target_length, tokenizer.model_max_length)
    logger.info(f"Model max length: {tokenizer.model_max_length}")
    logger.info(f"Max source length: {pcfg.max_source_length}")
    logger.info(f"Max target length: {pcfg.max_target_length}")

    logger.info(f"Training with model: {pcfg.model_name}")
    wandb.init(project=pcfg.wandb_project, config={"model": pcfg.model_name})

    logger.info(f"Preparing dataset")
    ds, tokenized = prepare_dataset(pcfg, args, tokenizer)

    logger.info(f"Creating trainer")
    trainer = create_trainer(
        ds=ds,
        tokenized=tokenized,
        tokenizer=tokenizer,
        model=model,
        args=args,
        pad_to_multiple_of=pcfg.pad_to_multiple_of,
        debug_tokens=pcfg.debug_tokens,
    )

    logger.info(f"Training")
    trainer.train()

    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if getattr(args, "push_to_hub", False) and getattr(args, "hub_model_id", None):
        logger.info(f"Pushing to hub")
        trainer.push_to_hub()


@click.group()
def cli():
    load_dotenv()
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )


@cli.command(name="estimate-lengths")
@click.option("--dataset-path", default="btseytlin/any2json", type=str)
@click.option("--model-name", default=DEFAULT_MODEL, type=str)
@click.option("--estimate-samples", default=2000, type=int)
def estimate_lengths_cmd(dataset_path: str, model_name: str, estimate_samples: int):
    estimate_token_lengths(dataset_path, model_name, estimate_samples)


@cli.command(
    name="train",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
@click.option("--dataset-path", default="btseytlin/any2json", type=str)
@click.option("--model-name", default=DEFAULT_MODEL, type=str)
@click.option("--max-source-length", default=None, type=int)
@click.option("--max-target-length", default=None, type=int)
@click.option("--drop-schema-proba", default=0.01, type=float)
@click.option("--schema-missing-token", default="[MISSING]", type=str)
@click.option("--input-aug", multiple=True, default=[], type=str)
@click.option("--output-aug", multiple=True, default=[], type=str)
@click.option("--debug-limit", default=None, type=int)
@click.option("--val-size", default=5000, type=int)
@click.option("--wandb-project", default="any2json", type=str)
@click.option("--pad-to-multiple-of", default=8, type=int)
@click.option("--debug-tokens", is_flag=True)
@click.option("--unsloth", is_flag=True)
def train_cmd(
    ctx: click.Context,
    dataset_path: str,
    model_name: str,
    max_source_length: int | None,
    max_target_length: int | None,
    drop_schema_proba: float,
    schema_missing_token: str,
    input_aug: tuple[str, ...],
    output_aug: tuple[str, ...],
    debug_limit: int | None,
    val_size: int,
    wandb_project: str,
    pad_to_multiple_of: int,
    debug_tokens: bool,
    unsloth: bool,
):
    parser = HfArgumentParser(TrainingArguments)
    hf_args_list = list(ctx.args)
    (args,) = parser.parse_args_into_dataclasses(hf_args_list)
    pcfg = PipelineConfig(
        dataset_path=dataset_path,
        model_name=model_name,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        drop_schema_proba=drop_schema_proba,
        schema_missing_token=schema_missing_token,
        input_aug=list(input_aug),
        output_aug=list(output_aug),
        debug_limit=debug_limit,
        val_size=val_size,
        wandb_project=wandb_project,
        pad_to_multiple_of=pad_to_multiple_of,
        debug_tokens=debug_tokens,
        unsloth=unsloth,
        hf_args=args,
    )
    if not args.output_dir:
        args.output_dir = "checkpoints"
    if args.group_by_length and not args.length_column_name:
        args.length_column_name = "length"
    if not args.report_to:
        args.report_to = ["wandb"]

    validate_pipeline_config(pcfg)
    validate_training_args(args)
    run_training(pcfg, args)


if __name__ == "__main__":
    cli()
