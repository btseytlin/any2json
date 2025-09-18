from dataclasses import dataclass, field
import json
from typing import Any, Callable
from datasets import Dataset
import random

import fastjsonschema

from any2json.data_engine.generators.vary_schema import VaryJSONSchemaGenerator
from any2json.training.constants import SCHEMA_MISSING_TOKEN
from any2json.utils import json_dumps_minified, logger


def aug_negative_sample(
    input_data: str,
    schema: str,
    output: str,
    augmentor: "Augmentor",
    rng: random.Random,
    dataset: Dataset | None = None,
    idx: int | None = None,
    max_attempts: int = 10,
    **kwargs,
) -> tuple[str, str, str]:
    if dataset is None or idx is None or schema == SCHEMA_MISSING_TOKEN:
        return input_data, schema, output

    indices = list(range(len(dataset)))
    indices.remove(idx)
    for _ in range(max_attempts):
        if not indices:
            break
        random_idx = rng.choice(indices)
        random_row = dataset[random_idx]
        random_schema = random_row["schema"]

        try:
            fastjsonschema.validate(json.loads(random_schema), json.loads(output))
            indices.remove(random_idx)
            continue
        except Exception as e:
            pass

        new_output = json.dumps(None)
        return input_data, random_schema, new_output
    logger.debug(f"Failed to find a negative sample after {max_attempts} attempts")
    return input_data, schema, output


def aug_drop_schema(
    input_data: str,
    schema: str,
    output: str,
    augmentor: "Augmentor",
    rng: random.Random,
    schema_missing_token: str = SCHEMA_MISSING_TOKEN,
    **kwargs,
) -> tuple[str, str, str]:
    return input_data, schema_missing_token, output


def aug_vary_schema_and_output(
    input_data: str,
    schema: str,
    output: str,
    augmentor: "Augmentor",
    rng: random.Random,
    **kwargs,
) -> tuple[str, str, str]:
    if schema == SCHEMA_MISSING_TOKEN:
        return input_data, schema, output

    vary_schema_generator = VaryJSONSchemaGenerator(rng=rng)
    vary_schema_generator.setup()

    source_data = json.loads(output)
    source_schema = json.loads(schema)

    source_schema, source_data, new_schema, new_data, changes = (
        vary_schema_generator.generate(
            source_data=source_data,
            source_schema=source_schema,
        )
    )

    new_schema = json_dumps_minified(new_schema)
    new_data = json_dumps_minified(new_data)

    logger.debug(f"Augmented schema, changes: {changes}")

    return input_data, new_schema, new_data


def get_random_json_dumps_kwargs(
    rng: random.Random,
) -> dict[str, Any]:
    separators = rng.choice([(",", ":"), (",", ": "), (", ", ": "), (", ", ":")])
    indent = rng.choice([None, 1, 2, 3, 4, 5])
    sort_keys = rng.choice([True, False])
    ensure_ascii = rng.choice([True, False])

    return dict(
        separators=separators,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
    )


def aug_vary_input_json_presentation(
    input_data: str,
    schema: str,
    output: str,
    augmentor: "Augmentor",
    rng: random.Random,
    **kwargs,
) -> tuple[str, str, str]:

    if input_data.startswith("{") or input_data.startswith("["):
        input_data_json = json.loads(input_data)
        input_data = json.dumps(input_data_json, **get_random_json_dumps_kwargs(rng))

    return input_data, schema, output


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
    **kwargs,
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
        aug_drop_schema: 0.1,
        aug_vary_schema_and_output: 0.8,
        aug_vary_input_json_presentation: 0.8,
        aug_corrupt_input: 0.8,
        aug_negative_sample: 0.05,
    }

    def apply(
        self,
        input_data: str,
        schema: str,
        output: str,
        rng: random.Random,
        dataset: Dataset | None = None,
        idx: int | None = None,
    ) -> tuple[str, str, str]:
        logger.debug(
            f"Applying augmentations.\n\nInput_data: {repr(input_data)}\nSchema: {repr(schema)}\nOutput: {repr(output)}"
        )
        for fn, proba in self.augmentations.items():
            if rng.random() < proba:
                logger.debug(f"Applying augmentation: {fn.__name__}")
                try:
                    input_data, schema, output = fn(
                        input_data,
                        schema,
                        output,
                        self,
                        rng,
                        dataset=dataset,
                        idx=idx,
                    )
                except Exception as e:
                    logger.debug(f"Error augmenting example: {e}")
        logger.debug(
            f"Augmented example.\n\nInput_data: {repr(input_data)}\nSchema: {repr(schema)}\nOutput: {repr(output)}"
        )
        return input_data, schema, output

    def __repr__(self) -> str:
        return f"Augmentor(augmentations={self.augmentations})"

    def __str__(self) -> str:
        return self.__repr__()
