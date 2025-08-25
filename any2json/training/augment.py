from dataclasses import dataclass, field
from typing import Any, Callable
from datasets import Dataset
import random

from any2json.training.constants import SCHEMA_MISSING_TOKEN
from any2json.utils import logger


def aug_drop_schema(
    input_data: str,
    schema: str,
    output: str,
    augmentor: "Augmentor",
    rng: random.Random,
    schema_missing_token: str = SCHEMA_MISSING_TOKEN,
):
    if rng.random() < augmentor.drop_schema_proba:
        return input_data, schema_missing_token, output
    return input_data, schema, output


def aug_vary_schema_and_output(
    input_data: str,
    schema: str,
    output: str,
    augmentor: "Augmentor",
    rng: random.Random,
):
    if rng.random() < augmentor.vary_schema_proba:
        return input_data, schema, output
    return input_data, schema, output


def aug_corrupt_input(
    input_data: str,
    schema: str,
    output: str,
    augmentor: "Augmentor",
    rng: random.Random,
):
    if rng.random() < augmentor.corrupt_input_proba:
        return input_data, schema, output
    return input_data, schema, output


@dataclass
class Augmentor:
    drop_schema_proba: float = 0.05
    vary_schema_proba: float = 0.2
    corrupt_input_proba: float = 0.2
    augmentations: list[Callable] = field(
        default_factory=lambda: [
            aug_drop_schema,
            aug_vary_schema_and_output,
            aug_corrupt_input,
        ]
    )

    def apply(
        self,
        input_data: str,
        schema: str,
        output: str,
        rng: random.Random,
    ) -> tuple[str, str, str]:
        for fn in self.augmentations:
            input_data, schema, output = fn(
                input_data,
                schema,
                output,
                self,
                rng,
            )
        return input_data, schema, output


def build_augment_fn(
    augmentor: Augmentor,
    seed: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)

    def augment_fn(batch: dict[str, Any]) -> dict[str, Any]:
        inputs, schemas, outputs = [], [], []
        for input_data, schema, output in zip(
            batch["input_data"],
            batch["schema"],
            batch["output"],
            strict=True,
        ):
            new_input_data, new_schema, new_output = augmentor.apply(
                input_data, schema, output, rng
            )
            logger.info(
                f"Augmented example:\n{new_input_data=}\n{new_schema=}\n{new_output=}"
            )
            inputs.append(new_input_data)
            schemas.append(new_schema)
            outputs.append(new_output)
        return {
            "input_data": inputs,
            "schema": schemas,
            "output": outputs,
            "meta": batch.get("meta"),
        }

    return augment_fn


def augment_dataset(
    dataset: Dataset,
    augmentor: Augmentor,
    seed: int = 0,
    num_proc: int = 8,
) -> Dataset:
    logger.info(f"Augmenting dataset with:\n{augmentor=} and {seed=}")
    augment_fn = build_augment_fn(
        augmentor=augmentor,
        seed=seed,
    )
    augmented = dataset.map(augment_fn, batched=True, num_proc=num_proc)
    return augmented
