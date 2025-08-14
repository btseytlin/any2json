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
import torch
from any2json.utils import configure_loggers, logger
from any2json.training.augment import build_augmentor, apply_augmentations
from any2json.training.utils import (
    format_example,
    load_hf_dataset,
    apply_debug_limit,
    make_group_split,
    percentile,
    EvalLoggerCallback,
    estimate_token_lengths,
)

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


@dataclass
class TrainingConfig:
    dataset_path: str
    model_name: str
    output_dir: str
    max_source_length: int
    max_target_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    auto_find_batch_size: bool
    learning_rate: float
    num_train_epochs: int
    warmup_ratio: float
    weight_decay: float
    seed: int
    gradient_accumulation_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    push_to_hub: bool
    hub_repo_id: str | None
    wandb_project: str
    bf16: bool
    fp16: bool
    use_cpu: bool
    drop_schema_proba: float
    schema_missing_token: str
    input_aug: list[str]
    output_aug: list[str]
    debug_limit: int | None
    gradient_checkpointing: bool
    predict_with_generate: bool
    val_size: int

    def validate(self) -> None:
        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16")
        if self.use_cpu and (self.fp16 or self.bf16):
            raise ValueError(
                "Mixed precision requires GPU; disable use_cpu or fp16/bf16"
            )
        if self.bf16 and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 requested but not supported on this hardware")
        if self.fp16 and not torch.cuda.is_available():
            raise ValueError("fp16 requested but CUDA is not available")
        if not 0 <= self.drop_schema_proba <= 1:
            raise ValueError("drop_schema_proba must be in [0, 1]")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if self.debug_limit is not None and self.debug_limit < 0:
            raise ValueError("debug_limit must be >= 0")
        for name in [
            "max_source_length",
            "max_target_length",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "num_train_epochs",
            "gradient_accumulation_steps",
            "logging_steps",
            "eval_steps",
            "save_steps",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.push_to_hub and not self.hub_repo_id:
            raise ValueError("hub_repo_id must be set when push_to_hub is True")


def build_tokenize_fn(
    tokenizer: AutoTokenizer, max_source_length: int, max_target_length: int
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
            a = a[:max_source_length]
            b = b[:max_target_length]
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


def augment_train_split(ds: DatasetDict, cfg: TrainingConfig) -> DatasetDict:
    aug = build_augmentor(
        drop_schema_proba=cfg.drop_schema_proba,
        schema_missing_token=cfg.schema_missing_token,
        input_aug_paths=cfg.input_aug,
        output_aug_paths=cfg.output_aug,
    )
    rng = random.Random(cfg.seed)

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
    ds: DatasetDict, tokenizer: AutoTokenizer, cfg: TrainingConfig
) -> DatasetDict:
    fn = build_tokenize_fn(tokenizer, cfg.max_source_length, cfg.max_target_length)
    train_tok = ds["train"].map(
        fn, batched=True, remove_columns=ds["train"].column_names
    )
    val_tok = ds["validation"].map(
        fn, batched=True, remove_columns=ds["validation"].column_names
    )
    return DatasetDict({"train": train_tok, "validation": val_tok})


def build_length_filter_fn(
    tokenizer: AutoTokenizer,
    max_source_length: int,
    max_target_length: int,
) -> Callable[[dict[str, Any]], list[bool]]:
    def pred(batch: dict[str, Any]) -> list[bool]:
        inputs = [
            format_example(i, s)
            for i, s in zip(batch["input_data"], batch["schema"], strict=True)
        ]
        src = tokenizer(inputs, padding=False, truncation=False, return_tensors=None)
        tgt = tokenizer(
            batch["output"], padding=False, truncation=False, return_tensors=None
        )
        return [
            (len(a) <= max_source_length) and (len(b) <= max_target_length)
            for a, b in zip(src["input_ids"], tgt["input_ids"], strict=True)
        ]

    return pred


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
    cfg: TrainingConfig,
):
    collator = CausalLMDataCollator(tokenizer=tokenizer)
    use_cpu = cfg.use_cpu
    local_bf16 = bool(cfg.bf16 and torch.cuda.is_bf16_supported())
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        auto_find_batch_size=cfg.auto_find_batch_size,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        use_cpu=use_cpu,
        bf16=local_bf16,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_repo_id,
        seed=cfg.seed,
        group_by_length=True,
        length_column_name="length",
        prediction_loss_only=True,
        torch_compile=True,
        weight_decay=cfg.weight_decay,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    return trainer


def run_training(cfg: TrainingConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    cfg.max_source_length = cfg.max_source_length or tokenizer.model_max_length // 2
    cfg.max_target_length = cfg.max_target_length or tokenizer.model_max_length // 2

    cfg.validate()
    logger.info(f"Training config: {cfg}")
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    wandb.init(project=cfg.wandb_project, config={"model": cfg.model_name})
    raw = load_hf_dataset(cfg.dataset_path)
    logger.info(f"Loaded {len(raw['train'])} train samples")

    if cfg.debug_limit:
        raw = apply_debug_limit(raw, cfg.debug_limit)
        logger.info(
            f"Applied debug limit: {cfg.debug_limit}, now {len(raw['train'])} train samples"
        )
    logger.info(f"Preparing splits with val size: {cfg.val_size}")
    ds = prepare_splits(raw, seed=cfg.seed, test_size=cfg.val_size)
    logger.info(f"Prepared splits with val size: {cfg.val_size}: {ds}")
    ds = augment_train_split(ds, cfg)
    logger.info(f"Augmented train split: {ds}")

    logger.info(f"Tokenizing splits")
    tokenized = tokenize_splits(ds, tokenizer, cfg)

    logger.info(
        f"Filtering tokenized data by length, max_source_length: {cfg.max_source_length}, max_target_length: {cfg.max_target_length}"
    )
    tokenized = filter_tokenized_splits_by_length(
        tokenized,
        max_source_length=cfg.max_source_length,
        max_target_length=cfg.max_target_length,
    )
    logger.info(f"Filtered tokenized datasets: {tokenized}")

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info(f"Creating trainer")
    trainer = create_trainer(tokenized, tokenizer, model, cfg)
    trainer.add_callback(EvalLoggerCallback(tokenizer, ds["validation"]))

    logger.info(f"Training")
    trainer.train()

    logger.info(f"Saving model to {cfg.output_dir}")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    if cfg.push_to_hub and cfg.hub_repo_id:
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


@cli.command(name="train")
@click.option("--dataset-path", default="btseytlin/any2json", type=str)
@click.option("--model-name", default=DEFAULT_MODEL, type=str)
@click.option("--output-dir", default="checkpoints", type=str)
@click.option("--max-source-length", default=None, type=int)
@click.option("--max-target-length", default=None, type=int)
@click.option("--per-device-train-batch-size", default=1, type=int)
@click.option("--per-device-eval-batch-size", default=1, type=int)
@click.option("--learning-rate", default=5e-5, type=float)
@click.option("--num-train-epochs", default=2, type=int)
@click.option("--warmup-ratio", default=0.03, type=float)
@click.option("--weight-decay", default=0.05, type=float)
@click.option("--seed", default=42, type=int)
@click.option("--gradient-accumulation-steps", default=8, type=int)
@click.option("--logging-steps", default=50, type=int)
@click.option("--eval-steps", default=500, type=int)
@click.option("--save-steps", default=1000, type=int)
@click.option("--push-to-hub", is_flag=True, default=False)
@click.option("--hub-repo-id", default=None, type=str)
@click.option("--wandb-project", default="any2json", type=str)
@click.option("--bf16", is_flag=True, default=False)
@click.option("--fp16", is_flag=True, default=False)
@click.option("--use-cpu", is_flag=True, default=False)
@click.option("--drop-schema-proba", default=0.01, type=float)
@click.option("--schema-missing-token", default="[MISSING]", type=str)
@click.option("--input-aug", multiple=True, default=[], type=str)
@click.option("--output-aug", multiple=True, default=[], type=str)
@click.option("--debug-limit", default=None, type=int)
@click.option("--gradient-checkpointing", is_flag=True, default=True)
@click.option("--predict-with-generate", is_flag=True, default=False)
@click.option("--val-size", default=5000, type=int)
@click.option("--auto-find-batch-size", is_flag=True, default=True)
@click.option("--weight-decay", default=0, type=float)
def train_cmd(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    max_source_length: int,
    max_target_length: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    num_train_epochs: int,
    warmup_ratio: float,
    weight_decay: float,
    seed: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    push_to_hub: bool,
    hub_repo_id: str | None,
    wandb_project: str,
    bf16: bool,
    fp16: bool,
    use_cpu: bool,
    drop_schema_proba: float,
    schema_missing_token: str,
    input_aug: tuple[str, ...],
    output_aug: tuple[str, ...],
    debug_limit: int,
    gradient_checkpointing: bool,
    predict_with_generate: bool,
    val_size: int,
    auto_find_batch_size: bool,
    weight_decay: float,
):
    cfg = TrainingConfig(
        dataset_path=dataset_path,
        model_name=model_name,
        output_dir=output_dir,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        seed=seed,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
        wandb_project=wandb_project,
        bf16=bf16,
        fp16=fp16,
        use_cpu=use_cpu,
        drop_schema_proba=drop_schema_proba,
        schema_missing_token=schema_missing_token,
        input_aug=list(input_aug),
        output_aug=list(output_aug),
        debug_limit=debug_limit,
        gradient_checkpointing=gradient_checkpointing,
        predict_with_generate=predict_with_generate,
        val_size=val_size,
        auto_find_batch_size=auto_find_batch_size,
        weight_decay=weight_decay,
    )
    run_training(cfg)


if __name__ == "__main__":
    cli()
