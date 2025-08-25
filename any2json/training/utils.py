import os
import random
from typing import Any, Callable
import json
import click
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset
import torch
from transformers import AutoTokenizer

from any2json.training.augment import (
    augment_dataset,
    Augmentor,
)
from any2json.training.constants import SCHEMA_MISSING_TOKEN, SYSTEM_PROMPT
from any2json.utils import logger
from any2json.grouping import train_test_split_groups


def resolve_pad_id(tokenizer: AutoTokenizer) -> int:
    return tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.unk_token_id


def format_example(
    input_data: str | dict,
    schema: str | dict | None = None,
    output: str | dict = "",
    missing_schema_token: str = SCHEMA_MISSING_TOKEN,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    if isinstance(input_data, dict):
        input_data = json.dumps(input_data, separators=(",", ":"), indent=None)

    if schema is None:
        schema = missing_schema_token
    if isinstance(schema, dict):
        schema = json.dumps(schema, separators=(",", ":"), indent=None)
    if isinstance(output, dict):
        output = json.dumps(output, separators=(",", ":"), indent=None)

    return f"{system_prompt}[SCHEMA]{schema}[INPUT]{input_data}[OUTPUT]{output}"


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


def ids_to_token_str(tok: AutoTokenizer, ids: list[int]) -> str:
    out: list[str] = []
    for t in ids:
        if t == -100:
            out.append("-100")
        else:
            out.append(tok.convert_ids_to_tokens([t], skip_special_tokens=False)[0])
    return " ".join(out)


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


def build_train_sequence(
    tokenizer: AutoTokenizer,
    prompt_ids: list[int],
    target_ids: list[int],
) -> tuple[list[int], list[int]]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    ids = [bos] + prompt_ids + target_ids + [eos]
    labels = ([-100] * (len(prompt_ids) + 1)) + target_ids + [eos]
    return ids, labels


def build_tokenize_fn(
    tokenizer: AutoTokenizer,
    debug: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        input_data = batch["input_data"]
        schema = batch["schema"]
        output = batch["output"]
        prompts = [
            format_example(i, s) for i, s in zip(input_data, schema, strict=True)
        ]
        enc_in = tokenizer(prompts, add_special_tokens=False)
        enc_out = tokenizer(output, add_special_tokens=False)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        lengths: list[int] = []
        for idx, (prompt_ids, target_ids) in enumerate(
            zip(enc_in["input_ids"], enc_out["input_ids"], strict=True)
        ):
            ids, lbs = build_train_sequence(tokenizer, prompt_ids, target_ids)
            input_ids.append(ids)
            labels.append(lbs)
            lengths.append(len(ids))
            if debug and idx == 0:
                logger.debug(f"DEBUG TOKENIZATION - Example {idx}:")
                logger.debug(f"  Input prompt: {repr(prompts[idx])}")
                logger.debug(f"  Input tokens: {prompt_ids}")
                logger.debug(f"  Input decoded: {repr(tokenizer.decode(prompt_ids))}")
                logger.debug(f"  Target tokens: {target_ids}")
                logger.debug(f"  Target decoded: {repr(tokenizer.decode(target_ids))}")
                logger.debug(f"  Output: {repr(batch['output'][idx])}")
                logger.debug(f"  EOS token: {eos}")
                logger.debug(f"  Final input_ids: {ids}")
                logger.debug(f"  Final labels: {lbs}")
                logger.debug(f"  Final length: {len(ids)}")
        return {"input_ids": input_ids, "labels": labels, "length": lengths}

    return tokenize


def pad_to_multiple(
    ids: list[list[int]], multiple: int, pad_id: int
) -> tuple[list[list[int]], list[list[int]]]:
    max_len = max(len(i) for i in ids)
    max_len = ((max_len + multiple - 1) // multiple) * multiple
    out = []
    attn = []
    for i in ids:
        pad = max(0, max_len - len(i))
        out.append(i + [pad_id] * pad)
        attn.append([1] * len(i) + [0] * pad)
    return out, attn


class CausalLMDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_id = resolve_pad_id(self.tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids, attention_mask = pad_to_multiple(
            [f["input_ids"] for f in features],
            self.pad_to_multiple_of,
            self.pad_id,
        )
        labels, _ = pad_to_multiple(
            [f["labels"] for f in features],
            self.pad_to_multiple_of,
            -100,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def encode_prompt_ids(
    tokenizer: AutoTokenizer, input_data: str, schema: str
) -> list[int]:
    enc = tokenizer(format_example(input_data, schema), add_special_tokens=False)
    return enc["input_ids"]


def encode_target_ids(tokenizer: AutoTokenizer, output: str) -> list[int]:
    enc = tokenizer(output, add_special_tokens=False)
    return enc["input_ids"]


def build_tokenized_length_filter_fn(
    max_sequence_length: int,
) -> Callable[[dict[str, Any]], list[bool]]:
    def pred(batch: dict[str, Any]) -> list[bool]:
        return [len(seq) <= max_sequence_length for seq in batch["input_ids"]]

    return pred


def filter_tokenized_splits_by_length(
    ds: DatasetDict,
    max_sequence_length: int,
    num_proc: int = 8,
) -> DatasetDict:
    pred = build_tokenized_length_filter_fn(max_sequence_length)
    train_f = ds["train"].filter(pred, batched=True, num_proc=num_proc)
    val_f = ds["validation"].filter(pred, batched=True, num_proc=num_proc)
    return DatasetDict({"train": train_f, "validation": val_f})


def process_raw_to_tokenized(
    dataset: Dataset,
    tokenize_fn: Callable[[dict[str, Any]], dict[str, Any]],
    filter_fn: Callable[[dict[str, Any]], bool],
    augmentor: Augmentor | None = None,
    seed: int = 0,
    num_proc: int = 8,
) -> Dataset:

    if augmentor:
        dataset = augment_dataset(
            dataset=dataset,
            augmentor=augmentor,
            seed=seed,
            num_proc=num_proc,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    filtered = tokenized.filter(filter_fn, batched=True, num_proc=num_proc)

    return filtered
