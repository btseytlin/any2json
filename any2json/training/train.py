import os
import random
from dataclasses import dataclass
from typing import Any, Callable

import click
from dotenv import load_dotenv
import wandb
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
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
)


@dataclass
class TrainingConfig:
    dataset_path: str
    model_name: str
    output_dir: str
    max_source_length: int
    max_target_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
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
    debug_limit: int
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
        if self.debug_limit < 0:
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


def estimate_token_lengths(dataset_path: str, model_name: str, samples: int) -> None:
    ds_all = load_hf_dataset(dataset_path)
    base = ds_all["train"] if "train" in ds_all else list(ds_all.values())[0]
    n = min(samples, len(base))
    rows = [base[i] for i in range(n)]
    tok = AutoTokenizer.from_pretrained(model_name)
    src = [
        len(tok(format_example(r["input_data"], r["schema"]))["input_ids"])
        for r in rows
    ]
    tgt = [len(tok(r["output"])["input_ids"]) for r in rows]
    click.echo(
        f"src p90={percentile(src, 0.9)} p95={percentile(src, 0.95)} p99={percentile(src, 0.99)}"
    )
    click.echo(
        f"tgt p90={percentile(tgt, 0.9)} p95={percentile(tgt, 0.95)} p99={percentile(tgt, 0.99)}"
    )


def build_tokenize_fn(
    tokenizer: AutoTokenizer, max_source_length: int, max_target_length: int
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        inputs = [
            format_example(i, s)
            for i, s in zip(batch["input_data"], batch["schema"], strict=True)
        ]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        labels = tokenizer(
            text_target=batch["output"],
            max_length=max_target_length,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize


class EvalLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer: AutoTokenizer, raw_eval_ds):
        self.tokenizer = tokenizer
        self.raw_eval_ds = raw_eval_ds
        self.table = wandb.Table(
            columns=["epoch", "step", "input", "schema", "target", "prediction"],
            log_mode="INCREMENTAL",
        )

    def on_evaluate(
        self, args, state, control, model=None, tokenizer=None, metrics=None, **kwargs
    ):
        if model is None:
            return
        k = min(8, len(self.raw_eval_ds))
        idx = random.sample(range(len(self.raw_eval_ds)), k)
        rows = [self.raw_eval_ds[i] for i in idx]
        prompts = [
            f"[SCHEMA]\n{r['schema']}\n[INPUT]\n{r['input_data']}\n[OUTPUT]\n"
            for r in rows
        ]
        device = next(model.parameters()).device
        toks = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=(getattr(self.tokenizer, "model_max_length", None) or 4096),
        )
        toks = {k2: v.to(device) for k2, v in toks.items()}
        out = model.generate(
            **toks,
            max_new_tokens=self.tokenizer.model_max_length // 4,
            do_sample=False,
            num_beams=1,
        )
        preds = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        for r, p in zip(rows, preds, strict=True):
            self.table.add_data(
                state.epoch,
                state.global_step,
                r["input_data"],
                r["schema"],
                r["output"],
                p,
            )
        wandb.log({"eval_examples": self.table}, step=state.global_step)


def prepare_splits(ds: DatasetDict, seed: int, size: int = 5000) -> DatasetDict:
    base = DatasetDict({"train": ds["train"]}) if "train" in ds else ds
    size = len(base["train"]) if "train" in base else 0
    test_size = min(size, size) if size > 5000 else max(1, size // 20)
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
    tokenizer: AutoTokenizer, max_source_length: int, max_target_length: int
) -> Callable[[dict[str, Any]], list[bool]]:
    def pred(batch: dict[str, Any]) -> list[bool]:
        inputs = [
            format_example(i, s)
            for i, s in zip(batch["input_data"], batch["schema"], strict=True)
        ]
        src = tokenizer(inputs, padding=False, truncation=False, return_tensors=None)
        tgt = tokenizer(
            text_target=batch["output"],
            padding=False,
            truncation=False,
            return_tensors=None,
        )
        return [
            (len(a) <= max_source_length) and (len(b) <= max_target_length)
            for a, b in zip(src["input_ids"], tgt["input_ids"], strict=True)
        ]

    return pred


def filter_splits_by_length(
    ds: DatasetDict, tokenizer: AutoTokenizer, cfg: TrainingConfig
) -> DatasetDict:
    pred = build_length_filter_fn(
        tokenizer, cfg.max_source_length, cfg.max_target_length
    )
    train_f = ds["train"].filter(pred, batched=True)
    val_f = ds["validation"].filter(pred, batched=True)
    return DatasetDict({"train": train_f, "validation": val_f})


def create_trainer(
    tokenized: DatasetDict,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    cfg: TrainingConfig,
) -> Seq2SeqTrainer:
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    use_cpu = cfg.use_cpu
    local_bf16 = bool(cfg.bf16 and torch.cuda.is_bf16_supported())
    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="steps",
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        predict_with_generate=cfg.predict_with_generate,
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
    )

    def metrics_fn(_):
        return {}

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metrics_fn,
    )
    return trainer


def run_training(cfg: TrainingConfig) -> None:
    cfg.validate()
    logger.info(f"Training config: {cfg}")
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    wandb.init(project=cfg.wandb_project, config={"model": cfg.model_name})
    raw = load_hf_dataset(cfg.dataset_path)
    raw = apply_debug_limit(raw, cfg.debug_limit)
    ds = prepare_splits(raw, seed=cfg.seed, size=cfg.val_size)
    ds = augment_train_split(ds, cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    ds = filter_splits_by_length(ds, tokenizer, cfg)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    tokenized = tokenize_splits(ds, tokenizer, cfg)
    trainer = create_trainer(tokenized, tokenizer, model, cfg)
    trainer.add_callback(EvalLoggerCallback(tokenizer, ds["validation"]))
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    if cfg.push_to_hub and cfg.hub_repo_id:
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
@click.option("--model-name", default="google/flan-t5-small", type=str)
@click.option("--estimate-samples", default=2000, type=int)
def estimate_lengths_cmd(dataset_path: str, model_name: str, estimate_samples: int):
    estimate_token_lengths(dataset_path, model_name, estimate_samples)


@cli.command(name="train")
@click.option("--dataset-path", default="btseytlin/any2json", type=str)
@click.option("--model-name", default="google/flan-t5-small", type=str)
@click.option("--output-dir", default="checkpoints", type=str)
@click.option("--max-source-length", default=2048, type=int)
@click.option("--max-target-length", default=1024, type=int)
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
@click.option("--debug-limit", default=0, type=int)
@click.option("--gradient-checkpointing", is_flag=True, default=True)
@click.option("--predict-with-generate", is_flag=True, default=False)
@click.option("--val-size", default=5000, type=int)
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
    )
    run_training(cfg)


if __name__ == "__main__":
    cli()
