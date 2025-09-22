import json
import pytest
import random
import fastjsonschema
from datasets import Dataset
from any2json.data_engine.generators.vary_schema import VaryJSONSchemaGenerator
from any2json.training.augment import Augmentor, aug_negative_sample
from any2json.training.constants import SCHEMA_MISSING_TOKEN
from any2json.utils import json_dumps_minified


class TestVaryJSONSchemaGenerator:
    def test_drop_fields_deterministic(self):
        generator = VaryJSONSchemaGenerator(drop_field_proba=0.1, rng=random.Random(42))
        generator.setup()

        source_schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]},
                "city": {"type": ["string", "null"]},
            },
        }

        new_schema, dropped = generator.drop_schema_keys_recursive(source_schema)
        expected_dropped = ["name"]
        assert set(dropped) == set(expected_dropped)

        expected_schema = {
            "type": "object",
            "properties": {
                "age": {"type": ["integer", "null"]},
                "city": {"type": ["string", "null"]},
            },
        }

        assert new_schema == expected_schema

    def test_drop_fields_nested_deterministic(self):
        generator = VaryJSONSchemaGenerator(
            drop_field_proba=0.4,
            rng=random.Random(35),
        )
        generator.setup()

        source_schema = {
            "type": "object",
            "properties": {
                "prop1": {"type": ["string", "null"]},
                "subobject": {
                    "type": "object",
                    "properties": {
                        "prop2": {"type": ["string", "null"]},
                        "prop3": {"type": ["integer", "null"]},
                        "prop4": {
                            "type": "subsubobject",
                            "properties": {
                                "prop8": {"type": ["string", "null"]},
                                "prop9": {"type": ["integer", "null"]},
                                "prop10": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
                "subarray": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prop5": {"type": ["string", "null"]},
                            "prop6": {"type": ["integer", "null"]},
                            "prop7": {"type": ["string", "null"]},
                        },
                    },
                },
            },
        }

        new_schema, dropped = generator.drop_schema_keys_recursive(source_schema)
        expected_dropped = [
            "subarray[].prop5",
            "subarray[].prop6",
            "subobject.prop2",
        ]
        assert set(dropped) == set(expected_dropped)

        expected_schema = {
            "type": "object",
            "properties": {
                "prop1": {"type": ["string", "null"]},
                "subobject": {
                    "type": "object",
                    "properties": {
                        "prop3": {"type": ["integer", "null"]},
                        "prop4": {
                            "type": "subsubobject",
                            "properties": {
                                "prop8": {"type": ["string", "null"]},
                                "prop9": {"type": ["integer", "null"]},
                                "prop10": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
                "subarray": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prop7": {"type": ["string", "null"]},
                        },
                    },
                },
            },
        }

        assert new_schema == expected_schema

    def test_add_fields_deterministic(self):
        generator = VaryJSONSchemaGenerator(rng=random.Random(42))
        generator.setup()
        generator.num_fields_to_add = 2

        source_schema = {
            "type": "object",
            "properties": {"existing": {"type": ["string", "null"]}},
        }

        new_schema, added = generator.add_schema_keys_recursive(source_schema)

        assert len(new_schema["properties"]) == len(source_schema["properties"]) + 2
        assert len(added) == 2
        assert "existing" in new_schema["properties"]

        for field_name, type_name in added:
            field_def = new_schema["properties"][field_name]
            assert field_def["type"][0] == type_name
            assert isinstance(field_def["type"], list)
            assert len(field_def["type"]) == 2
            assert "null" in field_def["type"]
            non_null_type = [t for t in field_def["type"] if t != "null"][0]
            assert non_null_type in ["string", "integer", "number", "boolean"]

    def test_add_fields_deterministic_nested(self):
        generator = VaryJSONSchemaGenerator(
            add_field_proba=0.8,
            rng=random.Random(35),
        )
        generator.setup()
        generator.num_fields_to_add = 2

        source_schema = {
            "type": "object",
            "properties": {
                "prop1": {"type": ["string", "null"]},
                "subobject": {
                    "type": "object",
                    "properties": {
                        "prop2": {"type": ["string", "null"]},
                        "prop3": {"type": ["integer", "null"]},
                        "prop4": {
                            "type": "object",
                            "properties": {
                                "prop8": {"type": ["string", "null"]},
                                "prop9": {"type": ["integer", "null"]},
                                "prop10": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
                "subarray": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prop5": {"type": ["string", "null"]},
                            "prop6": {"type": ["integer", "null"]},
                            "prop7": {"type": ["string", "null"]},
                        },
                    },
                },
            },
        }

        new_schema, added = generator.add_schema_keys_recursive(source_schema)
        expected_added = [
            ("law", "number"),
            ("among", "number"),
            ("subobject.day", "number"),
            ("subobject.happen", "string"),
            ("subobject.prop4.research", "boolean"),
            ("subarray[].stuff", "boolean"),
        ]
        assert set(added) == set(expected_added)

        expected_schema = {
            "type": "object",
            "properties": {
                "prop1": {"type": ["string", "null"]},
                "law": {"type": ["number", "null"]},
                "among": {"type": ["number", "null"]},
                "subobject": {
                    "type": "object",
                    "properties": {
                        "day": {"type": ["number", "null"]},
                        "happen": {"type": ["string", "null"]},
                        "prop2": {"type": ["string", "null"]},
                        "prop3": {"type": ["integer", "null"]},
                        "prop4": {
                            "type": "object",
                            "properties": {
                                "research": {"type": ["boolean", "null"]},
                                "prop8": {"type": ["string", "null"]},
                                "prop9": {"type": ["integer", "null"]},
                                "prop10": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
                "subarray": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "stuff": {"type": ["boolean", "null"]},
                            "prop5": {"type": ["string", "null"]},
                            "prop6": {"type": ["integer", "null"]},
                            "prop7": {"type": ["string", "null"]},
                        },
                    },
                },
            },
        }
        assert new_schema == expected_schema

    def test_change_types_deterministic(self):
        generator = VaryJSONSchemaGenerator(
            stringify_number_proba=0.9, rng=random.Random(42)
        )
        generator.setup()

        source_schema = {
            "type": "object",
            "properties": {
                "count": {"type": ["integer", "null"]},
                "score": {"type": ["number", "null"]},
                "name": {"type": ["string", "null"]},
            },
        }

        new_schema, changed = generator.change_schema_key_types(source_schema)
        expected_changes = {"count": "number_to_string", "score": "number_to_string"}
        assert changed == expected_changes

        expected_schema = {
            "type": "object",
            "properties": {
                "count": {"type": ["string", "null"]},
                "score": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
            },
        }

        assert new_schema == expected_schema

    def test_change_schema_key_types_nested_deterministic(self):
        generator = VaryJSONSchemaGenerator(
            stringify_number_proba=0.9, rng=random.Random(42)
        )
        generator.setup()

        source_schema = {
            "type": "object",
            "properties": {
                "count": {"type": ["integer", "null"]},
                "subobject": {
                    "type": "object",
                    "properties": {
                        "nested_number": {"type": ["number", "null"]},
                        "nested_int": {"type": ["integer", "null"]},
                        "nested_string": {"type": ["string", "null"]},
                    },
                },
                "subarray": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "array_number": {"type": ["number", "null"]},
                            "array_string": {"type": ["string", "null"]},
                        },
                    },
                },
            },
        }

        new_schema, changed = generator.change_schema_key_types(source_schema)

        expected_schema = {
            "type": "object",
            "properties": {
                "count": {"type": ["string", "null"]},
                "subobject": {
                    "type": "object",
                    "properties": {
                        "nested_number": {"type": ["string", "null"]},
                        "nested_int": {"type": ["string", "null"]},
                        "nested_string": {"type": ["string", "null"]},
                    },
                },
                "subarray": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "array_number": {"type": ["string", "null"]},
                            "array_string": {"type": ["string", "null"]},
                        },
                    },
                },
            },
        }
        expected_changes = {
            "count": "number_to_string",
            "subobject.nested_number": "number_to_string",
            "subobject.nested_int": "number_to_string",
            "subarray[].array_number": "number_to_string",
        }

        assert new_schema == expected_schema
        assert changed == expected_changes

    def test_coerce_value_string_exact(self):
        generator = VaryJSONSchemaGenerator()

        assert generator.coerce_value(123, ["string"]) == "123"
        assert generator.coerce_value(45.67, ["string"]) == "45.67"
        assert generator.coerce_value(True, ["string"]) == "True"
        assert generator.coerce_value(None, ["string"]) == "None"
        assert generator.coerce_value("already_string", ["string"]) == "already_string"

    def test_coerce_value_integer_exact(self):
        generator = VaryJSONSchemaGenerator()

        assert generator.coerce_value("42", ["integer"]) == 42
        assert generator.coerce_value("3.0", ["integer"]) == 3
        assert generator.coerce_value(3.0, ["integer"]) == 3

        with pytest.raises(ValueError):
            assert (
                generator.coerce_value(3.7, ["integer"]) == 3.7
            )  # Can't convert float with decimals
        with pytest.raises(ValueError):
            assert generator.coerce_value("not_a_number", ["integer"]) == "not_a_number"

        with pytest.raises(ValueError):
            assert generator.coerce_value(None, ["integer"]) is None

    def test_coerce_value_number_exact(self):
        generator = VaryJSONSchemaGenerator()

        assert generator.coerce_value("3.14", ["number"]) == 3.14
        assert generator.coerce_value("42", ["number"]) == 42.0
        assert generator.coerce_value(42, ["number"]) == 42
        with pytest.raises(ValueError):
            assert generator.coerce_value("not_a_number", ["number"]) == "not_a_number"
        with pytest.raises(ValueError):
            assert generator.coerce_value(None, ["number"]) is None

    def test_coerce_value_boolean_exact(self):
        generator = VaryJSONSchemaGenerator()

        assert generator.coerce_value("true", ["boolean"]) is True
        assert generator.coerce_value("TRUE", ["boolean"]) is True
        assert generator.coerce_value("false", ["boolean"]) is False
        assert generator.coerce_value("FALSE", ["boolean"]) is False
        assert generator.coerce_value("1", ["boolean"]) is True
        assert generator.coerce_value("0", ["boolean"]) is False
        assert generator.coerce_value(True, ["boolean"]) is True
        with pytest.raises(ValueError):
            assert (
                generator.coerce_value("maybe", ["boolean"]) == "maybe"
            )  # Can't convert
        with pytest.raises(ValueError):
            assert generator.coerce_value(None, ["boolean"]) is None

    def test_get_new_data_exact_transformation(self):
        generator = VaryJSONSchemaGenerator()

        input_data = {"name": "Alice", "age": 30, "score": 95.5}
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]},
                "score": {"type": ["number", "null"]},
            },
        }
        changes = {}

        result = generator.get_new_data(input_data, schema, changes)

        expected_result = {"name": "Alice", "age": 30, "score": 95.5}
        assert result == expected_result

    def test_get_new_data_with_type_conversion_exact(self):
        generator = VaryJSONSchemaGenerator()

        input_data = {"score": 95, "name": "test"}
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
            },
        }
        changes = {"changed_types": {"score": "number_to_string"}}

        result = generator.get_new_data(input_data, schema, changes)

        expected_result = {"score": "95", "name": "test"}
        assert result == expected_result

    def test_get_new_data_with_missing_fields_exact(self):
        generator = VaryJSONSchemaGenerator()
        generator.setup()

        input_data = {"existing": "value"}
        schema = {
            "type": "object",
            "properties": {
                "existing": {"type": ["string", "null"]},
                "new_field1": {"type": ["string", "null"]},
                "new_field2": {"type": ["integer", "null"]},
            },
        }
        changes = {}

        result = generator.get_new_data(input_data, schema, changes)

        expected_result = {
            "existing": "value",
            "new_field1": None,
            "new_field2": None,
        }
        assert result == expected_result

    def test_get_new_data_with_coercion_exact(self):
        generator = VaryJSONSchemaGenerator()
        generator.setup()

        input_data = {
            "str_to_int": "42",
            "float_to_int": 3.0,
            "str_to_bool": "true",
            "int_to_str": 123,
        }
        schema = {
            "type": "object",
            "properties": {
                "str_to_int": {"type": ["integer", "null"]},
                "float_to_int": {"type": ["integer", "null"]},
                "str_to_bool": {"type": ["boolean", "null"]},
                "int_to_str": {"type": ["string", "null"]},
            },
        }
        changes = {}

        result = generator.get_new_data(input_data, schema, changes)

        expected_result = {
            "str_to_int": 42,
            "float_to_int": 3,
            "str_to_bool": True,
            "int_to_str": "123",
        }
        assert result == expected_result

    def test_get_new_data_empty_properties_exact(self):
        generator = VaryJSONSchemaGenerator()
        generator.setup()

        input_data = {}
        schema = {"type": "object", "properties": {}}
        changes = {}

        with pytest.raises(ValueError, match="Transformed data has only null values"):
            generator.get_new_data(input_data, schema, changes)

    def test_get_new_data_all_null_values_raises_error_exact(self):
        generator = VaryJSONSchemaGenerator()
        generator.setup()

        input_data = {}
        schema = {
            "type": "object",
            "properties": {
                "field1": {"type": ["string", "null"]},
                "field2": {"type": ["integer", "null"]},
            },
        }
        changes = {}

        with pytest.raises(ValueError, match="Transformed data has only null values"):
            generator.get_new_data(input_data, schema, changes)

    def test_full_generate_cycle_deterministic(self):
        generator = VaryJSONSchemaGenerator(
            drop_field_proba=0.2,
            stringify_number_proba=1,
            add_field_proba=1.0,
            rng=random.Random(41),
        )
        generator.setup()
        generator.num_fields_to_add = 1

        source_data = {"name": "Alice", "age": 30, "score": 95.5}
        source_schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]},
                "score": {"type": ["number", "null"]},
            },
        }

        result = generator.generate(source_data, source_schema)

        assert len(result) == 5
        orig_schema, orig_data, new_schema, new_data, changes = result

        assert orig_schema == source_schema
        assert orig_data == source_data

        assert isinstance(new_schema, dict)
        assert isinstance(new_data, dict)
        assert isinstance(changes, dict)

        assert new_schema["type"] == ["object", "null"]
        assert "properties" in new_schema

        assert changes == {
            "dropped_fields": ["age"],
            "added_fields": [("decision", "string")],
            "changed_types": {"score": "number_to_string"},
        }

        assert new_data == {"name": "Alice", "decision": None, "score": "95.5"}
        assert new_schema == {
            "type": ["object", "null"],
            "properties": {
                "name": {"type": ["string", "null"]},
                "decision": {"type": ["string", "null"]},
                "score": {"type": ["string", "null"]},
            },
        }

        compiled = fastjsonschema.compile(new_schema)
        compiled(new_data)

    def test_generate_samples_deterministic(self):
        generator = VaryJSONSchemaGenerator(
            drop_field_proba=0.0,
            stringify_number_proba=0.0,
            add_field_proba=1.0,
            rng=random.Random(42),
        )
        generator.setup()
        generator.num_fields_to_add = 1

        source_data = {"field": "value"}
        source_schema = {
            "type": "object",
            "properties": {"field": {"type": ["string", "null"]}},
        }

        samples = generator.generate_samples(
            source_data=source_data, source_schema=source_schema, num_samples=2
        )

        assert len(samples) == 2

        for i, sample in enumerate(samples):
            assert len(sample) == 5
            orig_schema, orig_data, new_schema, new_data, changes = sample

            assert orig_schema == source_schema
            assert orig_data == source_data

            assert len(new_schema["properties"]) == 2
            assert "field" in new_schema["properties"]

            assert new_data["field"] == "value"

    def test_validate_source_data_invalid_schema_exact(self):
        generator = VaryJSONSchemaGenerator()

        with pytest.raises(AssertionError, match="Source schema must be an object"):
            generator.validate_source_data({}, {"type": "array"})

    def test_validate_source_data_invalid_data_exact(self):
        generator = VaryJSONSchemaGenerator()

        schema = {
            "type": "object",
            "properties": {"required": {"type": "string"}},
            "required": ["required"],
        }

        with pytest.raises(fastjsonschema.exceptions.JsonSchemaException):
            generator.validate_source_data({}, schema)

    def test_generate_with_unsupported_schema_exact(self):
        generator = VaryJSONSchemaGenerator()

        source_data = {"field": "value"}
        source_schema = {"type": "object"}  # No properties

        with pytest.raises(
            NotImplementedError, match="Generator only supports objects for now"
        ):
            generator.generate(source_data, source_schema)

    def test_get_state_returns_exact_config(self):
        generator = VaryJSONSchemaGenerator(
            drop_field_proba=0.3,
            stringify_number_proba=0.4,
            add_field_proba=0.5,
            num_fields_to_add=2,
        )

        expected_state = {
            "drop_field_proba": 0.3,
            "stringify_number_proba": 0.4,
            "add_field_proba": 0.5,
            "num_fields_to_add": 2,
        }

        state = generator.get_state()
        assert state == expected_state

    def test_json_schema_type_to_pydantic_type_exact(self):
        generator = VaryJSONSchemaGenerator()

        # Test single types
        assert generator.json_schema_type_to_pydantic_type("string") == str
        assert generator.json_schema_type_to_pydantic_type("integer") == int
        assert generator.json_schema_type_to_pydantic_type("number") == float
        assert generator.json_schema_type_to_pydantic_type("boolean") == bool
        assert generator.json_schema_type_to_pydantic_type("array") == list
        assert generator.json_schema_type_to_pydantic_type("object") == dict
        assert generator.json_schema_type_to_pydantic_type("null") == type(None)

        # Test list types (Union types)
        list_type = generator.json_schema_type_to_pydantic_type(["string", "null"])
        assert list_type is not None  # Should be a Union type

    def test_same_rng_different_results(self):
        rng = random.Random(42)
        source_data = {"name": "Alice", "age": 30, "score": 95.5}
        source_schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]},
                "score": {"type": ["number", "null"]},
            },
        }

        generator = VaryJSONSchemaGenerator(rng=rng)
        generator.setup()
        first_state = generator.get_state()
        _, _, first_new_schema, first_new_data, first_changes = generator.generate(
            source_data, source_schema
        )

        expected_num_fields = 1
        assert generator.num_fields_to_add == expected_num_fields
        assert isinstance(generator.fake, object)

        generator = VaryJSONSchemaGenerator(rng=rng)
        generator.setup()
        second_state = generator.get_state()
        _, _, second_new_schema, second_new_data, second_changes = generator.generate(
            source_data, source_schema
        )

        assert first_state != second_state
        assert first_new_schema != second_new_schema
        assert first_new_data != second_new_data
        assert first_changes != second_changes

        generator = VaryJSONSchemaGenerator(rng=rng)
        generator.setup()
        second_state = generator.get_state()
        _, _, second_new_schema, second_new_data, second_changes = generator.generate(
            source_data, source_schema
        )

        assert first_state != second_state
        assert first_new_schema != second_new_schema
        assert first_new_data != second_new_data
        assert first_changes != second_changes

    def test_different_rng_same_seed_same_results(self):
        seed = 43
        source_data = {"name": "Alice", "age": 30, "score": 95.5}
        source_schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]},
                "score": {"type": ["number", "null"]},
            },
        }

        generator = VaryJSONSchemaGenerator(rng=random.Random(seed))
        generator.setup()
        first_state = generator.get_state()
        _, _, first_new_schema, first_new_data, first_changes = generator.generate(
            source_data, source_schema
        )

        expected_num_fields = 2
        assert generator.num_fields_to_add == expected_num_fields
        assert isinstance(generator.fake, object)

        generator = VaryJSONSchemaGenerator(rng=random.Random(seed))
        generator.setup()
        second_state = generator.get_state()
        _, _, second_new_schema, second_new_data, second_changes = generator.generate(
            source_data, source_schema
        )

        assert first_state == second_state
        assert first_new_schema == second_new_schema
        assert first_new_data == second_new_data
        assert first_changes == second_changes

        generator = VaryJSONSchemaGenerator(rng=random.Random(seed + 1))
        generator.setup()
        second_state = generator.get_state()
        _, _, second_new_schema, second_new_data, second_changes = generator.generate(
            source_data, source_schema
        )

        assert first_state != second_state
        assert first_new_schema != second_new_schema
        assert first_new_data != second_new_data
        assert first_changes != second_changes


class TestAugmentor:
    def test_same_seed_same_results(self):
        seed = 40
        augmentor = Augmentor()

        input_data, schema, output = (
            "info: data",
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "info": {"type": ["string", "null"]},
                    },
                }
            ),
            json.dumps({"info": "data"}),
        )

        result = augmentor.apply(input_data, schema, output, random.Random(seed))

        expected_input_data, expected_schema, expected_output = (
            "_i_'nf!o: da|ta",
            json_dumps_minified(
                {
                    "type": ["object", "null"],
                    "properties": {
                        "info": {"type": ["string", "null"]},
                    },
                }
            ),
            json_dumps_minified({"info": "data"}),
        )

        assert result[0] == expected_input_data
        assert json.loads(result[1]) == json.loads(expected_schema)
        assert json.loads(result[2]) == json.loads(expected_output)

        next_result = augmentor.apply(input_data, schema, output, random.Random(seed))

        assert next_result == result


class TestAugNegativeSample:
    @pytest.fixture
    def mock_dataset(self):
        return Dataset.from_list(
            [
                {
                    "schema": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                            },
                        }
                    ),
                    "output": json.dumps({"name": "Alice", "age": 30}),
                },
                {
                    "schema": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "count": {"type": "integer"},
                            },
                        }
                    ),
                    "output": json.dumps({"title": "Test", "count": 5}),
                },
                {
                    "schema": json.dumps(
                        {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    ),
                    "output": json.dumps(["item1", "item2"]),
                },
            ]
        )

    def test_returns_unchanged_when_dataset_none(self):
        rng = random.Random(42)
        augmentor = Augmentor()

        input_data = "test input"
        schema = json.dumps({"type": "object"})
        output = json.dumps({"data": "test"})

        result = aug_negative_sample(
            input_data, schema, output, augmentor, rng, dataset=None, idx=0
        )

        assert result == (input_data, schema, output)

    def test_returns_unchanged_when_idx_none(self, mock_dataset):
        rng = random.Random(42)
        augmentor = Augmentor()

        input_data = "test input"
        schema = json.dumps({"type": "object"})
        output = json.dumps({"data": "test"})

        result = aug_negative_sample(
            input_data, schema, output, augmentor, rng, dataset=mock_dataset, idx=None
        )

        assert result == (input_data, schema, output)

    def test_returns_unchanged_when_schema_missing_token(self, mock_dataset):
        rng = random.Random(42)
        augmentor = Augmentor()

        input_data = "test input"
        schema = SCHEMA_MISSING_TOKEN
        output = json.dumps({"data": "test"})

        result = aug_negative_sample(
            input_data, schema, output, augmentor, rng, dataset=mock_dataset, idx=0
        )

        assert result == (input_data, schema, output)

    def test_finds_negative_sample_deterministic(self, mock_dataset):
        rng = random.Random(42)
        augmentor = Augmentor()

        idx = 0
        input_data = json.dumps({"name": "Alice", "age": 30})
        schema = mock_dataset[0]["schema"]
        output = mock_dataset[0]["output"]

        result = aug_negative_sample(
            input_data, schema, output, augmentor, rng, dataset=mock_dataset, idx=idx
        )

        result_input, result_schema, result_output = result

        assert result_input == input_data
        assert result_schema != schema
        assert result_output == json.dumps([])
        assert result_schema in [mock_dataset[1]["schema"], mock_dataset[2]["schema"]]

    def test_max_attempts_fallback(self):
        dataset_with_matching_schemas = Dataset.from_list(
            [
                {
                    "schema": json.dumps(
                        {"type": "object", "properties": {"name": {"type": "string"}}}
                    ),
                    "output": json.dumps({"name": "test"}),
                },
                {
                    "schema": json.dumps(
                        {"type": "object", "properties": {"name": {"type": "string"}}}
                    ),
                    "output": json.dumps({"name": "another"}),
                },
                {
                    "schema": json.dumps(
                        {"type": "object", "properties": {"name": {"type": "string"}}}
                    ),
                    "output": json.dumps({"name": "third"}),
                },
            ]
        )

        rng = random.Random(42)
        augmentor = Augmentor()

        input_data = json.dumps({"name": "Alice", "age": 30})
        schema = dataset_with_matching_schemas[0]["schema"]
        output = dataset_with_matching_schemas[0]["output"]

        result = aug_negative_sample(
            input_data,
            schema,
            output,
            augmentor,
            rng,
            dataset=dataset_with_matching_schemas,
            idx=0,
            max_attempts=3,
        )

        assert result == (input_data, schema, output)

    def test_with_augmentor_integration(self, mock_dataset):
        augmentor = Augmentor()
        augmentor.augmentations = {aug_negative_sample: 1.0}

        rng = random.Random(42)

        input_data = json.dumps({"name": "Alice", "age": 30})
        schema = mock_dataset[0]["schema"]
        output = mock_dataset[0]["output"]

        result = augmentor.apply(
            input_data, schema, output, rng, dataset=mock_dataset, idx=0
        )

        result_input, result_schema, result_output = result

        assert result_input == input_data
        assert result_schema != schema
        assert result_output == json.dumps([])
