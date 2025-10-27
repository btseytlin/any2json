import os
import random
from typing import Any, Callable
import json
import click
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

from any2json.training.constants import SCHEMA_MISSING_TOKEN, SYSTEM_PROMPT
from any2json.grouping import train_test_split_groups
from any2json.utils import json_dumps_minified


def resolve_pad_id(tokenizer: AutoTokenizer) -> int:
    return tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.unk_token_id


def format_example(
    input_data: str,
    schema: str | dict | None,
    output: str | dict | None = "",
    missing_schema_token: str = SCHEMA_MISSING_TOKEN,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    if schema is None:
        schema = missing_schema_token

    if isinstance(schema, dict):
        schema_str = json_dumps_minified(schema) if schema else "[MISSING]"
    else:
        schema_str = schema

    if output and isinstance(output, dict):
        output_str = json_dumps_minified(output) if output else "[MISSING]"
    else:
        output_str = output

    return f"{system_prompt}[SCHEMA]{schema_str}[INPUT]{input_data}[OUTPUT]{output_str}"


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
        dataset = load_from_disk(path_or_repo)
    else:
        dataset = load_dataset(path_or_repo)

    def load_meta(example):
        example["meta"] = (
            json.loads(example["meta"])
            if isinstance(example["meta"], str)
            else example["meta"]
        )
        example["meta"] = example["meta"] or {}
        return example

    dataset = dataset.map(load_meta, batched=False, num_proc=8)
    return dataset


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


def build_train_sequence(
    tokenizer: AutoTokenizer,
    prompt_ids: list[int],
    target_ids: list[int],
    pad_to: int | None = None,
) -> tuple[list[int], list[int]]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    ids = [bos] + prompt_ids + target_ids + [eos]
    labels = ([-100] * (len(prompt_ids) + 1)) + target_ids + [eos]

    if pad_to is not None:
        ids = ids + [tokenizer.pad_token_id] * (pad_to - len(ids))
        labels = labels + [-100] * (pad_to - len(labels))
    return ids, labels


def build_tokenize_fn(
    tokenizer: AutoTokenizer,
    tokenizer_kwargs: dict[str, Any] = {},
    pad_to: int | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        input_data = batch["input_data"]
        schema = batch["schema"]
        output = batch["output"]
        prompts = [
            format_example(i, s) for i, s in zip(input_data, schema, strict=True)
        ]
        enc_in = tokenizer(prompts, add_special_tokens=False, **tokenizer_kwargs)
        enc_out = tokenizer(output, add_special_tokens=False, **tokenizer_kwargs)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        lengths: list[int] = []
        for idx, (prompt_ids, target_ids) in enumerate(
            zip(enc_in["input_ids"], enc_out["input_ids"], strict=True)
        ):
            ids, lbs = build_train_sequence(
                tokenizer,
                prompt_ids,
                target_ids,
                pad_to=pad_to,
            )
            input_ids.append(ids)
            labels.append(lbs)
            lengths.append(len(ids))
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


class CausalLMDataCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        padding: str = "longest",
        max_length: int | None = None,
        pad_to_multiple_of: int = 8,
        return_tensors: str = "pt",
    ):
        assert (
            tokenizer.padding_side == "right"
        ), "Tokenizer must have padding_side == 'right' or the collator will not work correctly"
        super().__init__(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        no_label_features = [
            {k: v for k, v in f.items() if k != "labels"} for f in features
        ]
        batch = super().__call__(no_label_features)

        labels, _ = pad_to_multiple(
            [f["labels"] for f in features],
            self.pad_to_multiple_of,
            -100,
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


def build_tokenized_length_filter_fn(
    max_sequence_length: int | None,
) -> Callable[[dict[str, Any]], list[bool]]:
    def pred(batch: dict[str, Any]) -> list[bool]:
        if max_sequence_length is None:
            return [True for seq in batch["input_ids"]]
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
    num_proc: int = 8,
) -> Dataset:
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
    filtered = tokenized.filter(filter_fn, batched=True, num_proc=num_proc)
    return filtered


def prepare_stub_dataset(
    max_sequence_length: int,
    batch_size: int,
    tokenizer: AutoTokenizer,
) -> tuple[Any, Dataset]:
    """Prepare a fake dataset for testing batch sizes.

    It produces a dataset with `batch_size` samples where all inputs are of length `max_sequence_length`.
    """

    from any2json.training.dataset import AugmentTokenizeDataset

    stub_data = [
        {
            "input_data": '{"test": "value"}',
            "schema": '{"type": "object"}',
            "output": "output",
        }
    ] * batch_size

    raw_ds = Dataset.from_list(stub_data)

    tokenization_kwargs = {
        "pad_to": max_sequence_length,
    }

    tokenize_fn = build_tokenize_fn(tokenizer, **tokenization_kwargs)

    def pass_filter(batch: dict[str, Any]) -> list[bool]:
        return [True for seq in batch["input_ids"]]

    filter_fn = pass_filter

    train_dataset = AugmentTokenizeDataset.from_raw_dataset(
        dataset=raw_ds,
        tokenizer=tokenizer,
        tokenization_kwargs=tokenization_kwargs,
        filter_fn=filter_fn,
        dataloader_num_proc=1,
        augmentor=None,
    )

    eval_dataset = process_raw_to_tokenized(
        dataset=raw_ds,
        tokenize_fn=tokenize_fn,
        filter_fn=filter_fn,
        num_proc=1,
    )
    assert len(train_dataset) == batch_size
    assert len(eval_dataset) == batch_size
    for i in range(len(train_dataset)):
        assert train_dataset[i]["length"] == max_sequence_length
        assert len(train_dataset[i]["input_ids"]) == max_sequence_length
        assert eval_dataset[i]["length"] == max_sequence_length
        assert len(eval_dataset[i]["input_ids"]) == max_sequence_length

    return train_dataset, eval_dataset


def prepare_splits(ds: DatasetDict, seed: int, test_size: int = 5000) -> DatasetDict:
    base = DatasetDict({"train": ds["train"]}) if "train" in ds else ds
    size = len(base["train"]) if "train" in base else 0
    test_size = min(size, test_size) if size > test_size else max(1, size // 20)
    return make_group_split(base, test_size=test_size, seed=seed)
