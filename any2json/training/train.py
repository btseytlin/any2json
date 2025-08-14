import os
import random
from dataclasses import dataclass
from typing import Any, Callable

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
    format_example,
    load_hf_dataset,
    apply_debug_limit,
    make_group_split,
    EvalLoggerCallback,
    estimate_token_lengths,
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


def build_tokenize_fn(
    tokenizer: AutoTokenizer,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    eos = tokenizer.eos_token_id

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        prompts = [
            format_example(i, s)
            for i, s in zip(batch["input_data"], batch["schema"], strict=True)
        ]
        enc_in = tokenizer(prompts, add_special_tokens=False)
        enc_out = tokenizer(batch["output"], add_special_tokens=False)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        lengths: list[int] = []
        for a, b in zip(enc_in["input_ids"], enc_out["input_ids"], strict=True):
            ids = a + b + ([eos] if eos is not None else [])
            lbs = ([-100] * len(a)) + b + ([-100] if eos is not None else [])
            input_ids.append(ids)
            labels.append(lbs)
            lengths.append(len(ids))
        return {"input_ids": input_ids, "labels": labels, "length": lengths}

    return tokenize


def prepare_splits(ds: DatasetDict, seed: int, test_size: int = 5000) -> DatasetDict:
    base = DatasetDict({"train": ds["train"]}) if "train" in ds else ds
    size = len(base["train"]) if "train" in base else 0
    test_size = min(size, test_size) if size > test_size else max(1, size // 20)
    return make_group_split(base, test_size=test_size, seed=seed)


def augment_train_split(ds: DatasetDict, cfg: PipelineConfig, seed: int) -> DatasetDict:
    aug = build_augmentor(
        drop_schema_proba=cfg.drop_schema_proba,
        schema_missing_token=cfg.schema_missing_token,
        input_aug_paths=cfg.input_aug,
        output_aug_paths=cfg.output_aug,
    )
    rng = random.Random(seed)

    def map_fn(batch: dict[str, Any]) -> dict[str, Any]:
        inputs, schemas, outputs = [], [], []
        for i, s, o in zip(
            batch["input_data"], batch["schema"], batch["output"], strict=True
        ):
            ni, ns, no = apply_augmentations(i, s, o, aug, rng.random)
            inputs.append(ni)
            schemas.append(ns)
            outputs.append(no)
        return {
            "input_data": inputs,
            "schema": schemas,
            "output": outputs,
            "meta": batch.get("meta"),
        }

    train_aug = ds["train"].map(map_fn, batched=True)
    return DatasetDict({"train": train_aug, "validation": ds["validation"]})


def tokenize_splits(
    ds: DatasetDict, tokenizer: AutoTokenizer, cfg: PipelineConfig
) -> DatasetDict:
    fn = build_tokenize_fn(tokenizer)
    train_tok = ds["train"].map(
        fn, batched=True, remove_columns=ds["train"].column_names
    )
    val_tok = ds["validation"].map(
        fn, batched=True, remove_columns=ds["validation"].column_names
    )
    return DatasetDict({"train": train_tok, "validation": val_tok})


def build_tokenized_length_filter_fn(
    max_source_length: int, max_target_length: int
) -> Callable[[dict[str, Any]], list[bool]]:
    def pred(batch: dict[str, Any]) -> list[bool]:
        return [
            (len(src) <= max_source_length) and (len(lbl) <= max_target_length)
            for src, lbl in zip(batch["input_ids"], batch["labels"], strict=True)
        ]

    return pred


def filter_tokenized_splits_by_length(
    ds: DatasetDict, max_source_length: int, max_target_length: int
) -> DatasetDict:
    pred = build_tokenized_length_filter_fn(max_source_length, max_target_length)
    train_f = ds["train"].filter(pred, batched=True)
    val_f = ds["validation"].filter(pred, batched=True)
    return DatasetDict({"train": train_f, "validation": val_f})


class CausalLMDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int | None = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        input_ids, labels, attn = [], [], []
        for f in features:
            ids, lbs = f["input_ids"], f["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lbs + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def create_trainer(
    tokenized: DatasetDict,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    args: TrainingArguments,
):
    collator = CausalLMDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    return trainer


def run_training(pcfg: PipelineConfig, args: TrainingArguments) -> None:
    validate_pipeline_config(pcfg)
    validate_training_args(args)
    tokenizer = AutoTokenizer.from_pretrained(pcfg.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    pcfg.max_source_length = pcfg.max_source_length or tokenizer.model_max_length // 2
    pcfg.max_target_length = pcfg.max_target_length or tokenizer.model_max_length // 2
    logger.info(f"Training with model: {pcfg.model_name}")
    os.environ.setdefault("WANDB_PROJECT", pcfg.wandb_project)
    wandb.init(project=pcfg.wandb_project, config={"model": pcfg.model_name})
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
        tokenized,
        max_source_length=pcfg.max_source_length or tokenizer.model_max_length // 2,
        max_target_length=pcfg.max_target_length or tokenizer.model_max_length // 2,
    )
    logger.info(f"Filtered tokenized datasets: {tokenized}")

    model = AutoModelForCausalLM.from_pretrained(pcfg.model_name)
    model.config.use_cache = False
    if getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    logger.info(f"Creating trainer")
    if not args.output_dir:
        args.output_dir = "checkpoints"
    if args.group_by_length and not args.length_column_name:
        args.length_column_name = "length"
    if not args.report_to:
        args.report_to = ["wandb"]
    trainer = create_trainer(tokenized, tokenizer, model, args)
    trainer.add_callback(EvalLoggerCallback(tokenizer, ds["validation"]))

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
    )
    run_training(pcfg, args)


if __name__ == "__main__":
    cli()
