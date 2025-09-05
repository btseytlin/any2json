import pytest
import random
import fastjsonschema
from any2json.data_engine.generators.vary_schema import VaryJSONSchemaGenerator


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

        new_schema, dropped = generator.drop_schema_keys(source_schema)

        expected_schema = {
            "type": "object",
            "properties": {
                "age": {"type": ["integer", "null"]},
                "city": {"type": ["string", "null"]},
            },
        }
        expected_dropped = ["name"]

        assert new_schema == expected_schema
        assert set(dropped) == set(expected_dropped)

    def test_add_fields_deterministic(self):
        generator = VaryJSONSchemaGenerator(rng=random.Random(42))
        generator.setup()
        generator.num_fields_to_add = 2

        source_schema = {
            "type": "object",
            "properties": {"existing": {"type": ["string", "null"]}},
        }

        new_schema, added = generator.add_schema_keys(source_schema)

        assert len(new_schema["properties"]) == len(source_schema["properties"]) + 2
        assert len(added) == 2
        assert "existing" in new_schema["properties"]

        for field_name in added:
            field_def = new_schema["properties"][field_name]
            assert isinstance(field_def["type"], list)
            assert len(field_def["type"]) == 2
            assert "null" in field_def["type"]
            non_null_type = [t for t in field_def["type"] if t != "null"][0]
            assert non_null_type in ["string", "integer", "number", "boolean"]

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

        expected_schema = {
            "type": "object",
            "properties": {
                "count": {"type": ["string", "null"]},
                "score": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
            },
        }
        expected_changes = {"count": "number_to_string", "score": "number_to_string"}

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

    def test_get_new_data_required_field_missing_exact(self):
        generator = VaryJSONSchemaGenerator()
        generator.setup()

        input_data = {"optional": "value"}
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": ["string", "null"]},
                "optional": {"type": ["string", "null"]},
            },
            "required": ["required_field"],
        }
        changes = {}

        with pytest.raises(ValueError, match="Missing required field: required_field"):
            generator.get_new_data(input_data, schema, changes)

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
            drop_field_proba=0.2, stringify_number_proba=1, rng=random.Random(41)
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
            "added_fields": ["provide"],
            "changed_types": {"score": "number_to_string"},
        }

        assert new_data == {"name": "Alice", "provide": None, "score": "95.5"}
        assert new_schema == {
            "type": ["object", "null"],
            "properties": {
                "name": {"type": ["string", "null"]},
                "provide": {"type": ["number", "null"]},
                "score": {"type": ["string", "null"]},
            },
        }

        compiled = fastjsonschema.compile(new_schema)
        compiled(new_data)

    def test_generate_samples_deterministic(self):
        generator = VaryJSONSchemaGenerator(
            drop_field_proba=0.0,
            stringify_number_proba=0.0,
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
            drop_field_proba=0.3, stringify_number_proba=0.4, num_fields_to_add=2
        )

        expected_state = {
            "drop_field_proba": 0.3,
            "stringify_number_proba": 0.4,
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

    def test_setup_randomizes_num_fields_deterministic(self):
        generator = VaryJSONSchemaGenerator(rng=random.Random(43))
        generator.setup()

        expected_num_fields = 2
        assert generator.num_fields_to_add == expected_num_fields
        assert isinstance(generator.fake, object)
