import os
import random
from typing import Any

import click
from datasets import load_from_disk, load_dataset, DatasetDict
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb
from transformers.trainer_callback import TrainerCallback
from any2json.utils import logger
from any2json.grouping import train_test_split_groups


def format_example(input_data: str, schema: str) -> str:
    return f"[SCHEMA]\n{schema}\n[INPUT]\n{input_data}\n[OUTPUT]\n"


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    v = sorted(values)
    k = int((len(v) - 1) * q)
    return v[k]


def make_group_split(ds_dict: DatasetDict, test_size: int, seed: int) -> DatasetDict:
    groups = [s["meta"]["group"] for s in ds_dict["train"]]
    items = list(range(len(groups)))
    train_idx, test_idx, _, _ = train_test_split_groups(
        items, groups, test_size=test_size, random_state=seed
    )
    train_split = ds_dict["train"].select(train_idx)
    eval_split = ds_dict["train"].select(test_idx)
    return DatasetDict({"train": train_split, "validation": eval_split})


def load_hf_dataset(path_or_repo: str) -> DatasetDict:
    if os.path.isdir(path_or_repo):
        return load_from_disk(path_or_repo)
    return load_dataset(path_or_repo)


def apply_debug_limit(ds: DatasetDict, limit: int) -> DatasetDict:
    if limit <= 0 or "train" not in ds:
        return ds
    n = min(limit, len(ds["train"]))
    return DatasetDict({"train": ds["train"].select(range(n))})


class EvalLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer: AutoTokenizer, raw_eval_ds, num_examples: int = 3):
        self.tokenizer = tokenizer
        self.raw_eval_ds = raw_eval_ds
        self.table = wandb.Table(
            columns=[
                "epoch",
                "step",
                "input",
                "schema",
                "target",
                "prompt",
                "prediction",
            ],
            log_mode="INCREMENTAL",
        )
        self.num_examples = min(num_examples, len(raw_eval_ds))

    def sample_rows(self) -> list[dict[str, Any]]:
        idx = random.sample(range(len(self.raw_eval_ds)), self.num_examples)
        return [self.raw_eval_ds[i] for i in idx]

    def build_prompts(self, rows: list[dict[str, Any]]) -> list[str]:
        return [
            format_example(
                input_data=r["input_data"],
                schema=r["schema"],
            )
            for r in rows
        ]

    def generate_prediction_for_prompt(
        self, model: Any, toks: dict[str, Any], max_new_tokens: int | None = None
    ) -> str:
        max_new_tokens = max_new_tokens or self.tokenizer.model_max_length // 2
        out = model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        attn = toks.get("attention_mask")
        start = (
            int(attn.sum(dim=1).item())
            if attn is not None
            else int(toks["input_ids"].shape[1])
        )
        seq = out[0]
        return self.tokenizer.decode(seq[start:], skip_special_tokens=False)

    def generate_predictions(
        self, model: Any, prompts: list[str]
    ) -> tuple[list[str], list[str]]:
        model.eval()
        old_use_cache = model.config.use_cache
        model.config.use_cache = True
        device = next(model.parameters()).device
        inputs: list[str] = []
        preds: list[str] = []
        with torch.inference_mode():
            for p in tqdm(prompts, desc="Generating eval example predictions"):
                toks = self.tokenizer(
                    p,
                    return_tensors="pt",
                    truncation=False,
                    max_length=self.tokenizer.model_max_length // 2,
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                pred = self.generate_prediction_for_prompt(
                    model,
                    toks,
                    max_new_tokens=int(len(toks["input_ids"][0]) * 0.7),
                )
                inputs.append(p)
                preds.append(pred)
        model.config.use_cache = old_use_cache
        return inputs, preds

    def log_examples(
        self,
        state: Any,
        rows: list[dict[str, Any]],
        input_prompts: list[str],
        preds: list[str],
    ) -> None:
        for r, ip, p in zip(rows, input_prompts, preds, strict=True):
            self.table.add_data(
                state.epoch,
                state.global_step,
                r["input_data"],
                r["schema"],
                r["output"],
                ip,
                p,
            )
        wandb.log({"eval_examples": self.table})

    def on_evaluate(
        self, args, state, control, model=None, tokenizer=None, metrics=None, **kwargs
    ):
        logger.info(f"Running ´val example predictions callback")
        if model is None:
            return
        rows = self.sample_rows()
        prompts = self.build_prompts(rows)
        input_prompts, preds = self.generate_predictions(model, prompts)
        logger.info(
            f"Generated {len(input_prompts)} predictions.\nPrompts:\n{input_prompts}\nPreds:\n{preds}"
        )
        self.log_examples(state, rows, input_prompts, preds)


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
