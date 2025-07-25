import random
from any2json.containers import FromOtherFormatSample, Sample
from copy import deepcopy
from pydantic import BaseModel, create_model, ValidationError
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


class VaryJSONSchemaGenerator(SampleGenerator):
    """
    Generate synthetic JSON schemas and matching JSON chunks by randomly varying a given JSON schema.
    """

    def __init__(
        self,
        drop_field_proba: float = 0.2,
        stringify_number_proba: float = 0.3,
        num_fields_to_add: int | None = None,
    ):
        self.drop_field_proba = drop_field_proba
        self.stringify_number_proba = stringify_number_proba
        self.num_fields_to_add = num_fields_to_add

    def setup(self):
        self.fake = Faker()
        self.num_fields_to_add = random.randint(1, 3)

    def get_state(self) -> dict:
        return {
            "drop_field_proba": self.drop_field_proba,
            "stringify_number_proba": self.stringify_number_proba,
            "num_fields_to_add": self.num_fields_to_add,
        }

    def change_schema_key_types(self, schema: dict) -> tuple[dict, dict]:
        new_schema = deepcopy(schema)
        changed_types = {}
        for key, value in new_schema["properties"].items():
            if (
                value.get("type") in ["number", "integer"]
                or isinstance(value.get("type"), list)
                and value.get("type")[0] in ["integer"]
            ):
                if random.random() <= self.stringify_number_proba:
                    value["type"] = "string"
                    changed_types[key] = "number_to_string"
        return new_schema, changed_types

    def drop_schema_keys(self, schema: dict) -> tuple[dict, dict]:
        """
        Drops random keys from the schema.

        In the output, the dropped keys will be missing since they are no longer present in the schema even though they were present in the source data.
        """
        new_schema = deepcopy(schema)

        keys_to_drop = [
            key
            for key in list(new_schema["properties"].keys())
            if random.random() < self.drop_field_proba
        ]

        for key in keys_to_drop:
            if "properties" in new_schema and key in new_schema["properties"]:
                del new_schema["properties"][key]
            if "required" in new_schema and key in new_schema["required"]:
                new_schema["required"].remove(key)
        return new_schema, keys_to_drop

    def add_schema_keys(self, schema: dict) -> tuple[dict, list[str]]:
        """
        Adds new random keys to the schema.

        In the output, the expected values for the new keys will be null since they are missing in the source data.
        """
        new_schema = deepcopy(schema)

        keys_to_add = []

        for _ in range(self.num_fields_to_add):
            key = self.fake.word()
            type = self.fake.random_element(
                elements=("string", "integer", "number", "boolean")
            )
            new_schema["properties"][key] = {"type": [type, "null"]}
            keys_to_add.append(key)

        return new_schema, keys_to_add

    def get_new_schema(self, input_schema: dict) -> tuple[dict, dict]:
        new_schema = deepcopy(input_schema)

        new_schema, keys_to_drop = self.drop_schema_keys(new_schema)

        new_schema, changed_types = self.change_schema_key_types(new_schema)

        new_schema, keys_to_add = self.add_schema_keys(new_schema)

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
                if value == "number_to_string":
                    input_data[key] = str(input_data[key])

        fields: dict[str, Any] = {}
        required_fields = new_schema.get("required", [])

        for prop_name, prop_def in new_schema["properties"].items():
            pydantic_type = self.json_schema_type_to_pydantic_type(
                prop_def.get("type", "any")
            )
            if prop_name in required_fields:
                fields[prop_name] = (pydantic_type, ...)
            else:
                fields[prop_name] = (Union[pydantic_type, None], None)

        DynamicModel = create_model("DynamicModel", **fields)

        try:
            model_instance = DynamicModel(**input_data)
            transformed_data = model_instance.model_dump(exclude_unset=False)

            fastjsonschema.validate(new_schema, transformed_data)

            assert any(
                [v is not None for v in transformed_data.values()]
            ), "Transformed data has no non-null values"

            return transformed_data
        except ValidationError as e:
            raise ValueError(f"Input data could not be transformed to new schema: {e}")
        except fastjsonschema.exceptions.JsonSchemaException as e:
            raise ValueError(f"Transformed data failed schema validation: {e}")

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

    def validate_source_data(self, source_data: dict, source_schema: dict):
        assert isinstance(
            source_data, dict
        ), "Source data must be provided and be a dictionary"
        assert isinstance(
            source_schema, dict
        ), "Source schema must be provided and be a dictionary"

        assert source_schema["type"] == "object", "Source schema must be an object"
        fastjsonschema.validate(source_schema, source_data)

    def generate(
        self,
        source_data: dict,
        source_schema: dict,
    ) -> tuple[dict, dict, dict, dict, dict]:
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
