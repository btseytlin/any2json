from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Iterable


InputAugmentation = Callable[[str], str]
OutputAugmentation = Callable[[str], str]


def load_callable(path: str) -> Callable[..., object]:
    if ":" in path:
        module_path, func_name = path.split(":", 1)
    else:
        module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)


@dataclass
class Augmentor:
    drop_schema_proba: float
    schema_missing_token: str
    input_augmentations: list[InputAugmentation]
    output_augmentations: list[OutputAugmentation]


def build_augmentor(
    drop_schema_proba: float = 0.01,
    schema_missing_token: str = "[MISSING]",
    input_aug_paths: Iterable[str] | None = None,
    output_aug_paths: Iterable[str] | None = None,
) -> Augmentor:
    input_fns = [load_callable(p) for p in (input_aug_paths or [])]
    output_fns = [load_callable(p) for p in (output_aug_paths or [])]
    return Augmentor(
        drop_schema_proba=drop_schema_proba,
        schema_missing_token=schema_missing_token,
        input_augmentations=list(input_fns),
        output_augmentations=list(output_fns),
    )


def apply_augmentations(
    input_data: str,
    schema: str,
    output: str,
    augmentor: Augmentor,
    rng_callable: Callable[[], float] | None,
) -> tuple[str, str, str]:
    new_input = input_data
    new_output = output
    new_schema = schema
    for fn in augmentor.input_augmentations:
        new_input = fn(new_input)
    for fn in augmentor.output_augmentations:
        new_output = fn(new_output)
    if rng_callable is not None and rng_callable() < augmentor.drop_schema_proba:
        new_schema = augmentor.schema_missing_token
    return new_input, new_schema, new_output
