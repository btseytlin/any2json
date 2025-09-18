from dataclasses import dataclass
from typing import Any, Callable
import random
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer
from any2json.training.augment import Augmentor
from any2json.training.utils import build_tokenize_fn
from any2json.utils import logger


class AugmentTokenizeDataset(TorchDataset):

    def __init__(
        self,
        dataset: HFDataset,
        lengths: list[int],
        tokenizer: AutoTokenizer,
        tokenization_kwargs: dict[str, Any] = {},
        augmentor: Augmentor | None = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.lengths = lengths
        self.tokenizer = tokenizer
        self.tokenization_kwargs = tokenization_kwargs
        self.augmentor = augmentor
        self.seed = seed

        self.rng = random.Random(seed)

    @classmethod
    def from_raw_dataset(
        cls,
        dataset: HFDataset,
        tokenizer: AutoTokenizer,
        filter_fn: Callable[[dict[str, Any]], bool],
        tokenization_kwargs: dict[str, Any] = {},
        dataloader_num_proc: int = 8,
        **kwargs,
    ):

        tokenize_fn = build_tokenize_fn(tokenizer, **tokenization_kwargs)

        def tokenize_with_idx(batch):
            out = tokenize_fn(batch)
            out["row_idx"] = batch["row_idx"]
            return out

        raw_with_idx = dataset.add_column("row_idx", list(range(len(dataset))))
        tokenized_for_length = raw_with_idx.map(
            tokenize_with_idx,
            batched=True,
            remove_columns=raw_with_idx.column_names,
            num_proc=dataloader_num_proc,
        )
        filtered = tokenized_for_length.filter(
            filter_fn,
            batched=True,
            num_proc=dataloader_num_proc,
        )
        indices = list(filtered["row_idx"])  # type: ignore[index]
        lengths = list(filtered["length"])  # type: ignore[index]
        filtered_raw = dataset.select(indices)

        return cls(
            dataset=filtered_raw,
            lengths=lengths,
            tokenizer=tokenizer,
            tokenization_kwargs=tokenization_kwargs,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tokenize_fn = build_tokenize_fn(self.tokenizer, **self.tokenization_kwargs)
        row = self.dataset[idx]
        input_data = row["input_data"]
        schema = row["schema"]
        output = row["output"]
        if self.augmentor:
            input_data, schema, output = self.augmentor.apply(
                input_data,
                schema,
                output,
                self.rng,
                idx=idx,
                dataset=self.dataset,
            )

        batch = {"input_data": [input_data], "schema": [schema], "output": [output]}
        tokenized = tokenize_fn(batch)
        return {
            "input_ids": tokenized["input_ids"][0],
            "labels": tokenized["labels"][0],
            "length": self.lengths[int(idx)],
        }
