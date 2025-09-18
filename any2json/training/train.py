import os
from dataclasses import dataclass
from typing import Any
import json
import click
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.hf_argparser import HfArgumentParser
import torch
from any2json.training.augment import Augmentor
from any2json.utils import (
    configure_loggers,
    logger,
    try_minify_json_string,
)
from any2json.training.utils import (
    build_tokenized_length_filter_fn,
    load_hf_dataset,
    apply_debug_limit,
    make_group_split,
    build_tokenize_fn,
    CausalLMDataCollator,
    prepare_splits,
    prepare_stub_dataset,
    process_raw_to_tokenized,
)
from any2json.training.callbacks import (
    EvalLoggerCallback,
    DebugTokensCallback,
)
from any2json.training.dataset import AugmentTokenizeDataset

# DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_MODEL = "google/gemma-3-270m"


@dataclass
class PipelineConfig:
    dataset_path: str | None = None
    model_name: str | None = None
    max_sequence_length: int | None = 8192
    drop_schema_proba: float | None = None
    schema_missing_token: str | None = None
    debug_limit: int | None = None
    val_size: int | None = 5000
    wandb_project: str | None = None
    pad_to_multiple_of: int | None = None
    debug_tokens: bool | None = None
    unsloth: bool | None = None
    dataloader_num_proc: int | None = None
    augment: bool | None = None
    attn_implementation: str | None = None

    hf_args: TrainingArguments | None = None


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


def create_trainer(
    train_dataset: AugmentTokenizeDataset,
    eval_dataset: Dataset,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    args: TrainingArguments,
    pad_to_multiple_of: int = 8,
    debug_tokens: bool = False,
    max_sequence_length: int | None = None,
):
    collator = CausalLMDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_sequence_length,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.add_callback(
        EvalLoggerCallback(
            tokenizer=tokenizer,
            collator=collator,
            tokenized_eval_ds=eval_dataset,
            pad_to_multiple_of=pad_to_multiple_of,
            max_new_tokens=500,
        )
    )
    if debug_tokens:
        trainer.add_callback(DebugTokensCallback(tokenizer=tokenizer))
    return trainer


def prepare_dataset(
    pcfg: PipelineConfig,
    args: TrainingArguments,
    tokenizer: AutoTokenizer,
) -> tuple[AugmentTokenizeDataset, Dataset]:
    raw = load_hf_dataset(pcfg.dataset_path)

    if pcfg.debug_limit:
        raw = apply_debug_limit(raw, pcfg.debug_limit)
        logger.info(
            f"Applied debug limit: {pcfg.debug_limit}, now {len(raw['train'])} train samples"
        )

    logger.info(f"Loaded {len(raw['train'])} train samples")

    def preprocess_on_load(item: dict[str, Any]) -> dict[str, Any]:
        item["output"] = try_minify_json_string(item["output"])
        item["schema"] = try_minify_json_string(item["schema"])
        return item

    logger.info("Minifying JSON schemas and outputs on load")

    raw["train"] = raw["train"].map(
        preprocess_on_load, batched=False, num_proc=pcfg.dataloader_num_proc
    )

    ds = prepare_splits(raw, args.seed, pcfg.val_size)
    logger.info(f"Prepared splits: {ds}")

    raw_train = ds["train"]

    tokenize_fn = build_tokenize_fn(tokenizer)
    filter_fn = build_tokenized_length_filter_fn(pcfg.max_sequence_length)

    augmentor = Augmentor() if pcfg.augment else None

    logger.info("Preparing train dataset")

    train_dataset = AugmentTokenizeDataset.from_raw_dataset(
        dataset=raw_train,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        dataloader_num_proc=pcfg.dataloader_num_proc,
        augmentor=augmentor,
        seed=args.seed,
    )

    logger.info("Preparing validation dataset")

    val_tokenized = process_raw_to_tokenized(
        dataset=ds["validation"],
        tokenize_fn=tokenize_fn,
        filter_fn=filter_fn,
        num_proc=pcfg.dataloader_num_proc,
    )
    logger.info("Prepared datasets")
    return train_dataset, val_tokenized


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

    tokenizer.padding_side = "right"

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if pcfg.max_sequence_length:
        if (
            not tokenizer.model_max_length
            or tokenizer.model_max_length != pcfg.max_sequence_length
        ):
            if tokenizer.model_max_length < pcfg.max_sequence_length:
                raise ValueError(
                    f"Model can not support max sequence length {pcfg.max_sequence_length}, maximum is {tokenizer.model_max_length}"
                )
            tokenizer.model_max_length = pcfg.max_sequence_length
            logger.warning(
                f"Model max length {tokenizer.model_max_length} forcefully set to {pcfg.max_sequence_length}"
            )
    return model, tokenizer


def run_training(pcfg: PipelineConfig, args: TrainingArguments) -> None:
    os.environ.setdefault("WANDB_PROJECT", pcfg.wandb_project)
    os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "False")

    logger.info(f"Pipeline config: {pcfg}")

    if pcfg.unsloth:
        try:
            import unsloth
        except ImportError:
            logger.warning("Unsloth is not installed. Not using it")
            pcfg.unsloth = False

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else device
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model and tokenizer")
    model, tokenizer = prepare_model_and_tokenizer(pcfg, args)
    model.to(device)

    pcfg.max_sequence_length = pcfg.max_sequence_length or tokenizer.model_max_length
    pcfg.max_sequence_length = min(pcfg.max_sequence_length, tokenizer.model_max_length)
    logger.info(f"Model max length: {tokenizer.model_max_length}")
    logger.info(f"Max sequence length: {pcfg.max_sequence_length}")

    logger.info(f"Training with model: {pcfg.model_name}")
    wandb.init(project=pcfg.wandb_project, config={"model": pcfg.model_name})

    logger.info(f"Preparing dataset")
    train_dataset, eval_dataset = prepare_dataset(pcfg, args, tokenizer)

    logger.info(f"Creating trainer")
    trainer = create_trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model=model,
        args=args,
        pad_to_multiple_of=pcfg.pad_to_multiple_of,
        debug_tokens=pcfg.debug_tokens,
        max_sequence_length=pcfg.max_sequence_length,
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
@click.pass_context
@click.option("--dataset-path", default="btseytlin/any2json", type=str)
@click.option("--model-name", default=DEFAULT_MODEL, type=str)
@click.option("--estimate-samples", default=2000, type=int)
def estimate_lengths_cmd(
    ctx: click.Context,
    dataset_path: str,
    model_name: str,
    estimate_samples: int,
):
    parser = HfArgumentParser(TrainingArguments)
    hf_args_list = list(ctx.args)
    (args,) = parser.parse_args_into_dataclasses(hf_args_list)
    pcfg = PipelineConfig(
        dataset_path=dataset_path,
        model_name=model_name,
        debug_limit=estimate_samples,
        hf_args=args,
    )
    pcfg.max_sequence_length = None
    _, tokenizer = prepare_model_and_tokenizer(pcfg, args)

    train_dataset, _ = prepare_dataset(pcfg, args, tokenizer)

    src = [len(r["input_ids"]) for r in train_dataset]
    tgt = [len([i for i in r["labels"] if i != -100]) for r in train_dataset]
    total = [src[i] + tgt[i] for i in range(len(src))]
    click.echo(
        f"Source sequences: p50={np.quantile(src, 0.5)} p90={np.quantile(src, 0.9)} p95={np.quantile(src, 0.95)} p99={np.quantile(src, 0.99)}"
    )
    click.echo(
        f"Completions: p50={np.quantile(tgt, 0.5)} p90={np.quantile(tgt, 0.9)} p95={np.quantile(tgt, 0.95)} p99={np.quantile(tgt, 0.99)}"
    )
    click.echo(
        f"Total: p50={np.quantile(total, 0.5)} p90={np.quantile(total, 0.9)} p95={np.quantile(total, 0.95)} p99={np.quantile(total, 0.99)}"
    )


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
@click.option("--debug-limit", default=None, type=int)
@click.option("--val-size", default=5000, type=int)
@click.option("--wandb-project", default="any2json", type=str)
@click.option("--pad-to-multiple-of", default=8, type=int)
@click.option("--debug-tokens", is_flag=True)
@click.option("--unsloth", is_flag=True)
@click.option("--dataloader-num-proc", default=8, type=int)
@click.option("--augment/--no-augment", default=True)
@click.option("--attn-implementation", default="eager", type=str)
def train_cmd(
    ctx: click.Context,
    dataset_path: str,
    model_name: str,
    max_sequence_length: int | None,
    drop_schema_proba: float,
    schema_missing_token: str,
    debug_limit: int | None,
    val_size: int,
    wandb_project: str,
    pad_to_multiple_of: int,
    debug_tokens: bool,
    unsloth: bool,
    dataloader_num_proc: int,
    augment: bool,
    attn_implementation: str,
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


def try_training(
    pcfg: PipelineConfig,
    args: TrainingArguments,
    batch_size: int,
) -> None:
    args.per_device_train_batch_size = batch_size
    args.per_device_eval_batch_size = batch_size
    args.max_steps = 2
    args.eval_steps = 1
    args.save_steps = 1000
    args.logging_steps = 1
    args.output_dir = "/tmp/batch_size_test"
    args.report_to = []
    args.dataloader_num_workers = 0

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else device

    model, tokenizer = prepare_model_and_tokenizer(pcfg, args)
    model.to(device)

    train_dataset, eval_dataset = prepare_stub_dataset(
        pcfg.max_sequence_length,
        args.per_device_train_batch_size,
        tokenizer,
    )

    trainer = create_trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model=model,
        args=args,
        pad_to_multiple_of=8,
        debug_tokens=False,
    )

    trainer.train()

    del trainer
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
@click.option("--model-name", default=DEFAULT_MODEL, type=str)
@click.option("--max-sequence-length", required=True, type=int)
@click.option("--max-per-device-train-batch-size", default=16, type=int)
@click.option("--batch-size-step", default=2, type=int)
def find_batch_size(
    ctx: click.Context,
    model_name: str,
    max_sequence_length: int,
    max_per_device_train_batch_size: int,
    batch_size_step: int,
):
    parser = HfArgumentParser(TrainingArguments)
    hf_args_list = list(ctx.args)
    (args,) = parser.parse_args_into_dataclasses(hf_args_list)
    pcfg = PipelineConfig(
        dataset_path=None,
        model_name=model_name,
        max_sequence_length=max_sequence_length,
        drop_schema_proba=None,
        schema_missing_token=None,
        debug_limit=None,
        val_size=None,
        wandb_project=None,
        pad_to_multiple_of=None,
        debug_tokens=None,
        unsloth=None,
        dataloader_num_proc=None,
        augment=None,
        attn_implementation=None,
        hf_args=args,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else device
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model and tokenizer")
    model, tokenizer = prepare_model_and_tokenizer(pcfg, args)
    model.to(device)

    pcfg.max_sequence_length = pcfg.max_sequence_length or tokenizer.model_max_length
    pcfg.max_sequence_length = min(pcfg.max_sequence_length, tokenizer.model_max_length)
    logger.info(f"Model max length: {tokenizer.model_max_length}")
    logger.info(f"Max sequence length: {pcfg.max_sequence_length}")

    batch_size_options = [max_per_device_train_batch_size]
    while True:
        new_batch_size = batch_size_options[-1] - batch_size_step
        if new_batch_size < batch_size_step:
            new_batch_size = batch_size_options[-1] - 1
        if new_batch_size < 1:
            break
        batch_size_options.append(new_batch_size)

    for batch_size in batch_size_options:
        logger.info(f"Trying batch size: {batch_size}")
        try:
            try_training(pcfg, args, batch_size)
            break
        except Exception as e:
            logger.error(f"Error training with batch size {batch_size}: {e}")
            continue
    else:
        raise RuntimeError("Training failed for all batch sizes")

    logger.info(f"Largest batch size found: {batch_size}")


if __name__ == "__main__":
    cli()
