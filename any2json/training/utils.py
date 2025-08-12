import os

from datasets import load_from_disk, load_dataset, DatasetDict

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
