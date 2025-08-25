from dataclasses import dataclass, field
import json
from typing import Any, Callable
from datasets import Dataset
import random

from any2json.data_engine.generators.vary_schema import VaryJSONSchemaGenerator
from any2json.training.constants import SCHEMA_MISSING_TOKEN
from any2json.utils import logger


def aug_drop_schema(
    input_data: str,
    schema: str,
    output: str,
    augmentor: "Augmentor",
    rng: random.Random,
    schema_missing_token: str = SCHEMA_MISSING_TOKEN,
) -> tuple[str, str, str]:
    return input_data, schema_missing_token, output


def aug_vary_schema_and_output(
    input_data: str,
    schema: str,
    output: str,
    augmentor: "Augmentor",
    rng: random.Random,
) -> tuple[str, str, str]:
    if schema == SCHEMA_MISSING_TOKEN:
        return input_data, schema, output

    vary_schema_generator = VaryJSONSchemaGenerator(rng=rng)
    vary_schema_generator.setup()
    try:
        source_schema, source_data, new_schema, new_data, changes = (
            vary_schema_generator.generate(
                source_data=json.loads(output),
                source_schema=json.loads(schema),
            )
        )
    except NotImplementedError:
        return input_data, schema, output

    new_schema = json.dumps(new_schema)
    new_data = json.dumps(new_data)

    return input_data, new_schema, new_data


# def get_random_json_dumps_kwargs(
#     rng: random.Random,
# ) -> dict[str, Any]:
#     separators = rng.choice([(",", ":"), (",", ": "), (", ", ": "), (", ", ":")])
#     indent = rng.choice([None, 1, 2, 3, 4, 5])
#     sort_keys = rng.choice([True, False])
#     ensure_ascii = rng.choice([True, False])

#     return dict(
#         separators=separators,
#         indent=indent,
#         sort_keys=sort_keys,
#         ensure_ascii=ensure_ascii,
#     )


# def aug_vary_json_presentation(
#     input_data: str,
#     schema: str,
#     output: str,
#     augmentor: "Augmentor",
#     rng: random.Random,
# ) -> tuple[str, str, str]:
#     if (
#         rng.random() > augmentor.vary_json_presentation_proba
#         or schema == SCHEMA_MISSING_TOKEN
#     ):
#         return input_data, schema, output

#     schema = json.loads(schema)
#     output = json.loads(output)

#     schema = json.dumps(schema, **get_random_json_dumps_kwargs(rng))
#     output = json.dumps(output, **get_random_json_dumps_kwargs(rng))

#     return input_data, schema, output


def aug_corrupt_input(
    input_data: str,
    schema: str,
    output: str,
    augmentor: "Augmentor",
    rng: random.Random,
    corruption_symbols: list[str] = [
        "!",
        "\t",
        "|",
        "-",
        "—",
        "_",
        ".",
        "'",
        '"',
        " ",
        "_",
    ],
    max_num_corruptions: int = 5,
) -> tuple[str, str, str]:
    """Insert a few junk symbols into the input data"""
    num_corruptions = rng.randint(1, max_num_corruptions)

    for _ in range(num_corruptions):
        corruption_symbol = rng.choice(corruption_symbols)
        input_data = list(input_data)
        input_data.insert(
            rng.choice(range(len(input_data))),
            corruption_symbol,
        )
        input_data = "".join(input_data)
    return input_data, schema, output


class Augmentor:
    augmentations: dict[Callable, float] = {
        aug_drop_schema: 0.05,
        aug_vary_schema_and_output: 0.2,
        aug_corrupt_input: 0.1,
    }

    def apply(
        self,
        input_data: str,
        schema: str,
        output: str,
        rng: random.Random,
    ) -> tuple[str, str, str]:
        logger.info(
            f"Applying augmentations.\n\nInput_data: {input_data}\nSchema: {schema}\nOutput: {output}"
        )
        for fn, proba in self.augmentations.items():
            if rng.random() < proba:
                logger.info(f"Applying augmentor: {fn.__name__}")
                input_data, schema, output = fn(
                    input_data,
                    schema,
                    output,
                    self,
                    rng,
                )
        logger.info(
            f"Augmented example.\n\nInput_data: {input_data}\nSchema: {schema}\nOutput: {output}"
        )
        return input_data, schema, output

    def __repr__(self) -> str:
        return f"Augmentor(augmentations={self.augmentations})"

    def __str__(self) -> str:
        return self.__repr__()


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
