import random
from any2json.containers import FromOtherFormatSample, Sample
from copy import deepcopy
from typing import Any, Type, Union, Callable
import fastjsonschema
import xml.etree.ElementTree as ET
import csv
import io
import yaml
import toml
from faker import Faker
from dataclasses import dataclass

from any2json.data_engine.generators.base import SampleGenerator
from any2json.schema_utils import to_supported_json_schema


def get_schema_keys_to_drop_recursive(
    rng: random.Random,
    drop_field_proba: float,
    current_schema: dict,
    current_path: str = "",
) -> list[str]:
    if not isinstance(current_schema, dict):
        return []

    dropped_paths = []

    if "properties" in current_schema:
        keys_to_drop = [
            key
            for key in list(current_schema["properties"].keys())
            if rng.random() < drop_field_proba
        ]

        for key in current_schema["properties"].keys():
            if key in keys_to_drop:
                path = f"{current_path}.{key}" if current_path else key
                dropped_paths.append(path)
            else:
                child_dropped_paths = get_schema_keys_to_drop_recursive(
                    rng,
                    drop_field_proba,
                    current_schema["properties"][key],
                    f"{current_path}.{key}" if current_path else key,
                )
                dropped_paths.extend(child_dropped_paths)

    if "items" in current_schema:
        items_dropped_paths = get_schema_keys_to_drop_recursive(
            rng,
            drop_field_proba,
            current_schema["items"],
            f"{current_path}[]",
        )
        dropped_paths.extend(items_dropped_paths)

    return dropped_paths


def drop_schema_keys_recursive(
    current_schema: dict,
    dropped_paths: list[str],
    current_path: str = "",
) -> dict:
    if not isinstance(current_schema, dict):
        return current_schema

    result_schema = deepcopy(current_schema)

    if "properties" in result_schema:
        this_path_keys = [
            key
            for key in result_schema["properties"].keys()
            if f"{current_path}.{key}" in dropped_paths or key in dropped_paths
        ]

        for key in this_path_keys:
            del result_schema["properties"][key]

        for key in list(result_schema["properties"].keys()):
            result_schema["properties"][key] = drop_schema_keys_recursive(
                result_schema["properties"][key],
                dropped_paths,
                f"{current_path}.{key}" if current_path else key,
            )

    if "items" in result_schema:
        result_schema["items"] = drop_schema_keys_recursive(
            result_schema["items"],
            dropped_paths,
            f"{current_path}[]",
        )

    return result_schema


def get_schema_keys_to_add_recursive(
    rng: random.Random,
    fake: Faker,
    add_field_proba: float,
    num_fields_to_add: int,
    current_schema: dict,
    current_path: str = "",
) -> list[tuple[str, dict]]:
    if not isinstance(current_schema, dict):
        return []

    keys_to_add = []

    if "properties" in current_schema:
        if rng.random() < add_field_proba:
            for _ in range(rng.randint(1, num_fields_to_add)):
                key = rng.choice(fake.words())
                type_name = rng.choice(["string", "integer", "number", "boolean"])
                field_def = {"type": [type_name, "null"]}
                path = f"{current_path}.{key}" if current_path else key
                keys_to_add.append((path, field_def))

        for key in current_schema["properties"].keys():
            child_keys_to_add = get_schema_keys_to_add_recursive(
                rng,
                fake,
                add_field_proba,
                num_fields_to_add,
                current_schema["properties"][key],
                f"{current_path}.{key}" if current_path else key,
            )
            keys_to_add.extend(child_keys_to_add)

    if "items" in current_schema:
        items_keys_to_add = get_schema_keys_to_add_recursive(
            rng,
            fake,
            add_field_proba,
            num_fields_to_add,
            current_schema["items"],
            f"{current_path}[]",
        )
        keys_to_add.extend(items_keys_to_add)

    return keys_to_add


def add_schema_keys_recursive(
    current_schema: dict,
    keys_to_add: list[tuple[str, dict]],
    current_path: str = "",
) -> dict:
    if not isinstance(current_schema, dict):
        return current_schema

    result_schema = deepcopy(current_schema)

    if "properties" in result_schema:
        for path, field_def in keys_to_add:
            if current_path == "":
                if "." not in path and "[]" not in path:
                    result_schema["properties"][path] = field_def
            else:
                expected_prefix = f"{current_path}."
                if path.startswith(expected_prefix):
                    remaining_path = path[len(expected_prefix) :]
                    if "." not in remaining_path and "[]" not in remaining_path:
                        result_schema["properties"][remaining_path] = field_def

        for key in list(result_schema["properties"].keys()):
            result_schema["properties"][key] = add_schema_keys_recursive(
                result_schema["properties"][key],
                keys_to_add,
                f"{current_path}.{key}" if current_path else key,
            )

    if "items" in result_schema:
        result_schema["items"] = add_schema_keys_recursive(
            result_schema["items"],
            keys_to_add,
            f"{current_path}[]",
        )

    return result_schema


class VaryJSONSchemaGenerator(SampleGenerator):
    """
    Generate synthetic JSON schemas and matching JSON chunks by randomly varying a given JSON schema.
    """

    def __init__(
        self,
        drop_field_proba: float = 0.2,
        stringify_number_proba: float = 0.3,
        add_field_proba: float = 0.3,
        num_fields_to_add: int | None = None,
        rng: random.Random | None = None,
    ):
        self.drop_field_proba = drop_field_proba
        self.stringify_number_proba = stringify_number_proba
        self.add_field_proba = add_field_proba
        self.num_fields_to_add = num_fields_to_add
        self.rng = rng or random.Random()

    def setup(self):
        self.fake = Faker()
        self.fake.seed_instance(self.rng.randint(0, 1000000))
        self.num_fields_to_add = self.rng.randint(1, 3)

    def get_state(self) -> dict:
        return {
            "drop_field_proba": self.drop_field_proba,
            "stringify_number_proba": self.stringify_number_proba,
            "add_field_proba": self.add_field_proba,
            "num_fields_to_add": self.num_fields_to_add,
        }

    def change_schema_key_types(self, schema: dict) -> tuple[dict, dict]:
        new_schema = deepcopy(schema)
        changed_types = {}
        for key, value in new_schema["properties"].items():
            if (
                value.get("type") in ["number", "integer"]
                or isinstance(value.get("type"), list)
                and value.get("type")[0] in ["number", "integer"]
            ):
                if self.rng.random() <= self.stringify_number_proba:
                    value["type"] = ["string", "null"]
                    changed_types[key] = "number_to_string"
        return new_schema, changed_types

    def drop_schema_keys_recursive(self, schema: dict) -> tuple[dict, list[str]]:
        """
        Drops random keys from the schema recursively at any level.
        Returns xpath-style indicators of dropped keys for corresponding JSON data.
        """
        new_schema = deepcopy(schema)

        dropped_paths = get_schema_keys_to_drop_recursive(
            self.rng, self.drop_field_proba, new_schema
        )

        new_schema = drop_schema_keys_recursive(new_schema, dropped_paths)

        return new_schema, dropped_paths

    def add_schema_keys_recursive(self, schema: dict) -> tuple[dict, list[str]]:
        """
        Adds new random keys to the schema recursively at any level.
        Returns xpath-style indicators of added keys for corresponding JSON data.
        """
        new_schema = deepcopy(schema)

        keys_to_add_with_defs = get_schema_keys_to_add_recursive(
            self.rng,
            self.fake,
            self.add_field_proba,
            self.num_fields_to_add,
            new_schema,
        )

        new_schema = add_schema_keys_recursive(new_schema, keys_to_add_with_defs)
        added_paths_and_types = [
            (path, field_def["type"][0]) for path, field_def in keys_to_add_with_defs
        ]

        return new_schema, added_paths_and_types

    def get_new_schema(self, input_schema: dict) -> tuple[dict, dict]:
        new_schema = deepcopy(input_schema)

        new_schema, keys_to_drop = self.drop_schema_keys_recursive(new_schema)

        new_schema, changed_types = self.change_schema_key_types(new_schema)

        new_schema, keys_to_add = self.add_schema_keys_recursive(new_schema)

        changes = {}
        if keys_to_drop:
            changes["dropped_fields"] = keys_to_drop
        if keys_to_add:
            changes["added_fields"] = keys_to_add
        if changed_types:
            changes["changed_types"] = changed_types

        return new_schema, changes

    def get_new_data(self, input_data: dict, new_schema: dict, changes: dict) -> dict:
        if changes.get("changed_types"):
            for key, value in changes["changed_types"].items():
                if value == "number_to_string" and key in input_data:
                    input_data[key] = str(input_data[key])

        properties: dict[str, Any] = new_schema["properties"]
        required_fields = set(new_schema.get("required", []))

        transformed: dict[str, Any] = {}
        for prop_name, prop_def in properties.items():
            value = input_data.get(prop_name, None)
            t = prop_def["type"]
            types = t if isinstance(t, list) else [t]
            if value is None:
                coerced = None
            else:
                coerced = self.coerce_value(value, types)
            if prop_name in required_fields and coerced is None:
                raise ValueError(f"Missing required field: {prop_name}")
            transformed[prop_name] = coerced

        compiled_schema = fastjsonschema.compile(
            new_schema,
            detailed_exceptions=False,
        )
        compiled_schema(transformed)

        if not transformed or all(v is None for v in transformed.values()):
            raise ValueError("Transformed data has only null values")
        return transformed

    def json_schema_type_to_pydantic_type(
        self, json_type: Union[str, list]
    ) -> Type[Any]:
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
            "any": Any,
        }

        if isinstance(json_type, list):
            pydantic_types = []
            for item_type in json_type:
                pydantic_types.append(type_map.get(item_type, Any))
            return Union[tuple(pydantic_types)]
        else:
            return type_map.get(json_type, Any)

    def coerce_value(self, value: Any, types: list[str]) -> Any:
        if "string" in types:
            return str(value)
        if "integer" in types:
            if isinstance(value, float):
                if float(value).is_integer():
                    return int(value)
                else:
                    raise ValueError(f"Cannot convert float to integer: {value}")
            if isinstance(value, str):
                return int(float(value))
            if isinstance(value, int):
                return value
        if "number" in types:
            if isinstance(value, (int, float)):
                return value
            elif isinstance(value, str):
                return float(value)
        if "boolean" in types:
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                s = value.strip().lower()
                if s in {"true", "1", "True"}:
                    return True
                if s in {"false", "0", "False"}:
                    return False
        raise ValueError(f"Cannot convert value to {types}: {value}")

    def validate_source_data(self, source_data: dict, source_schema: dict):
        assert isinstance(
            source_data, dict
        ), "Source data must be provided and be a dictionary"
        assert isinstance(
            source_schema, dict
        ), "Source schema must be provided and be a dictionary"

        assert source_schema["type"] == "object", "Source schema must be an object"
        compiled_schema = fastjsonschema.compile(
            source_schema,
            detailed_exceptions=False,
        )
        compiled_schema(source_data)

    def generate(
        self,
        source_data: dict,
        source_schema: dict,
    ) -> tuple[dict, dict, dict, dict, dict]:

        if "properties" not in source_schema:
            raise NotImplementedError("Generator only supports objects for now")

        new_schema, changes = self.get_new_schema(source_schema)

        new_data = self.get_new_data(source_data, new_schema, changes)

        new_schema = to_supported_json_schema(new_schema)

        return source_schema, source_data, new_schema, new_data, changes

    def generate_samples(
        self,
        source_data: dict = None,
        source_schema: dict = None,
        num_samples: int = 1,
    ) -> list[tuple[dict, dict, dict, dict, dict]]:
        self.validate_source_data(source_data, source_schema)
        return [self.generate(source_data, source_schema) for _ in range(num_samples)]
