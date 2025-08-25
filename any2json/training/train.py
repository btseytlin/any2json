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
from any2json.training.augment import Augmentor
from any2json.utils import configure_loggers, logger
from any2json.training.utils import (
    build_tokenized_length_filter_fn,
    load_hf_dataset,
    apply_debug_limit,
    make_group_split,
    estimate_token_lengths,
    build_tokenize_fn,
    CausalLMDataCollator,
    process_raw_to_tokenized,
)
from any2json.training.callbacks import (
    EvalLoggerCallback,
    DebugTokensCallback,
    RetokenizationCallback,
)

# DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_MODEL = "google/gemma-3-270m"


@dataclass
class PipelineConfig:
    dataset_path: str
    model_name: str
    max_sequence_length: int
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
    dataloader_num_proc: int
    augment: bool
    attn_implementation: str


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


def create_trainer(
    tokenized: DatasetDict,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    args: TrainingArguments,
    pad_to_multiple_of: int = 8,
    debug_tokens: bool = False,
    retokenization_callback: RetokenizationCallback = None,
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
            tokenizer=tokenizer,
            collator=collator,
            tokenized_eval_ds=tokenized["validation"],
            pad_to_multiple_of=pad_to_multiple_of,
            max_new_tokens=500,
        )
    )
    if debug_tokens:
        trainer.add_callback(DebugTokensCallback(tokenizer=tokenizer))
    if retokenization_callback:
        trainer.add_callback(retokenization_callback)
    return trainer


def prepare_dataset(
    pcfg: PipelineConfig,
    args: TrainingArguments,
    tokenizer: AutoTokenizer,
) -> tuple[DatasetDict, RetokenizationCallback]:
    raw = load_hf_dataset(pcfg.dataset_path)
    logger.info(f"Loaded {len(raw['train'])} train samples")

    if pcfg.debug_limit:
        raw = apply_debug_limit(raw, pcfg.debug_limit)
        logger.info(
            f"Applied debug limit: {pcfg.debug_limit}, now {len(raw['train'])} train samples"
        )

    ds = prepare_splits(raw, args.seed, pcfg.val_size)
    logger.info(f"Prepared splits: {ds}")

    raw_train = ds["train"]

    tokenize_fn = build_tokenize_fn(tokenizer, debug=pcfg.debug_tokens)
    filter_fn = build_tokenized_length_filter_fn(pcfg.max_sequence_length)

    augmentor = None
    if pcfg.augment:
        augmentor = Augmentor()

    logger.info("Processing initial tokenized dataset")
    train_tokenized = process_raw_to_tokenized(
        dataset=raw_train,
        tokenize_fn=tokenize_fn,
        filter_fn=filter_fn,
        augmentor=augmentor,
        seed=args.seed,
        num_proc=pcfg.dataloader_num_proc,
    )

    logger.info("Tokenizing validation dataset")

    val_tokenized = process_raw_to_tokenized(
        dataset=ds["validation"],
        tokenize_fn=tokenize_fn,
        filter_fn=filter_fn,
        augmentor=None,
        num_proc=pcfg.dataloader_num_proc,
    )

    tokenized = DatasetDict({"train": train_tokenized, "validation": val_tokenized})

    callback = RetokenizationCallback(
        raw_train_dataset=raw_train,
        tokenize_fn=tokenize_fn,
        filter_fn=filter_fn,
        augmentor=augmentor,
        seed=args.seed,
        num_proc=pcfg.dataloader_num_proc,
    )

    logger.info(f"Prepared datasets: {tokenized}")
    return tokenized, callback


def prepare_model_and_tokenizer(
    pcfg: PipelineConfig,
    args: TrainingArguments,
) -> tuple[AutoModelForCausalLM, AutoTokenizer | None]:
    if pcfg.unsloth:
        from unsloth import FastModel

        assert "unsloth" in pcfg.model_name, "Must use an unsloth model with --unsloth"

        model, tokenizer = FastModel.from_pretrained(
            pcfg.model_name,
            full_finetuning=True,
            use_gradient_checkpointing="unsloth",
            max_seq_length=pcfg.max_sequence_length,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pcfg.model_name,
            attn_implementation=pcfg.attn_implementation,
        )
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

    pcfg.max_sequence_length = pcfg.max_sequence_length or tokenizer.model_max_length
    pcfg.max_sequence_length = min(pcfg.max_sequence_length, tokenizer.model_max_length)
    logger.info(f"Model max length: {tokenizer.model_max_length}")
    logger.info(f"Max sequence length: {pcfg.max_sequence_length}")

    logger.info(f"Training with model: {pcfg.model_name}")
    wandb.init(project=pcfg.wandb_project, config={"model": pcfg.model_name})

    logger.info(f"Preparing dataset")
    tokenized, retokenization_callback = prepare_dataset(pcfg, args, tokenizer)

    logger.info(f"Creating trainer")
    trainer = create_trainer(
        tokenized=tokenized,
        tokenizer=tokenizer,
        model=model,
        args=args,
        pad_to_multiple_of=pcfg.pad_to_multiple_of,
        debug_tokens=pcfg.debug_tokens,
        retokenization_callback=retokenization_callback,
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
@click.option("--max-sequence-length", default=None, type=int)
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
@click.option("--dataloader-num-proc", default=8, type=int)
@click.option("--augment", is_flag=True)
@click.option("--attn-implementation", default="sdpa", type=str)
def train_cmd(
    ctx: click.Context,
    dataset_path: str,
    model_name: str,
    max_sequence_length: int | None,
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
    dataloader_num_proc: int,
    augment: bool,
):
    parser = HfArgumentParser(TrainingArguments)
    hf_args_list = list(ctx.args)
    (args,) = parser.parse_args_into_dataclasses(hf_args_list)
    pcfg = PipelineConfig(
        dataset_path=dataset_path,
        model_name=model_name,
        max_sequence_length=max_sequence_length,
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
        dataloader_num_proc=dataloader_num_proc,
        augment=augment,
        attn_implementation=attn_implementation,
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
