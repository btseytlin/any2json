import pytest
import json
from unittest.mock import patch, MagicMock
from any2json.data_engine.helpers import (
    extract_explicit_subschemas,
    extract_implicit_subschemas,
    extract_subschemas_from_schema,
    extract_schema_from_ref,
    fetch_schema_from_url,
    validate_schema,
    expand_refs_in_schema,
)
from any2json.database.models import JsonSchema


class TestExtractExplicitSubschemas:
    def test_extract_from_defs(self):
        schema = {
            "type": "object",
            "properties": {"person": {"$ref": "#/$defs/Person"}},
            "$defs": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
        }

        subschemas = extract_explicit_subschemas(schema)

        assert len(subschemas) == 2
        paths = [path for _, path in subschemas]
        assert "$defs.Person" in paths
        assert "$defs.Address" in paths

        person_schema = [s for s, p in subschemas if p == "$defs.Person"][0]
        assert person_schema["type"] == "object"
        assert "name" in person_schema["properties"]

    def test_extract_from_definitions(self):
        schema = {
            "type": "object",
            "definitions": {
                "User": {
                    "type": "object",
                    "properties": {"username": {"type": "string"}},
                }
            },
        }

        subschemas = extract_explicit_subschemas(schema)

        assert len(subschemas) == 1
        assert subschemas[0][1] == "definitions.User"

    def test_no_explicit_subschemas(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        subschemas = extract_explicit_subschemas(schema)

        assert len(subschemas) == 0


class TestExtractImplicitSubschemas:
    def test_extract_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                }
            },
        }

        subschemas = extract_implicit_subschemas(schema)

        assert len(subschemas) == 1
        assert subschemas[0][1] == "properties.user"
        assert subschemas[0][0]["type"] == "object"

    def test_extract_array_items(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
            },
        }

        subschemas = extract_implicit_subschemas(schema)

        assert len(subschemas) == 1
        assert subschemas[0][1] == "items"
        assert subschemas[0][0]["type"] == "object"

    def test_extract_nested_arrays(self):
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"value": {"type": "number"}},
                        },
                    },
                }
            },
        }

        subschemas = extract_implicit_subschemas(schema)

        assert len(subschemas) >= 3
        paths = [path for _, path in subschemas]
        assert "properties.data" in paths
        assert "properties.data.items" in paths
        assert "properties.data.items.items" in paths

    def test_extract_additional_properties(self):
        schema = {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
            },
        }

        subschemas = extract_implicit_subschemas(schema)

        assert len(subschemas) == 1
        assert subschemas[0][1] == "additionalProperties"

    def test_deeply_nested_structure(self):
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {"value": {"type": "string"}},
                                    },
                                }
                            },
                        }
                    },
                }
            },
        }

        subschemas = extract_implicit_subschemas(schema)

        assert len(subschemas) >= 4
        paths = [path for _, path in subschemas]
        assert "properties.level1" in paths
        assert "properties.level1.properties.level2" in paths
        assert "properties.level1.properties.level2.properties.level3" in paths

    def test_skip_primitive_types(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            },
        }

        subschemas = extract_implicit_subschemas(schema)

        assert len(subschemas) == 0


class TestExtractSubschemasFromSchema:
    def test_extract_from_schema_entity(self):
        schema_content = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                }
            },
            "$defs": {
                "User": {"type": "object", "properties": {"id": {"type": "integer"}}}
            },
        }

        schema_entity = JsonSchema(
            id=1, content=schema_content, is_synthetic=False, meta={"source": "test"}
        )

        new_schemas = extract_subschemas_from_schema(schema_entity)

        assert len(new_schemas) >= 2

        for new_schema in new_schemas:
            assert new_schema.parent_schema_id == 1
            assert new_schema.meta["source"] == "extracted_subschema"
            assert new_schema.meta["original_schema_id"] == 1
            assert "extraction_path" in new_schema.meta

    def test_meta_preservation(self):
        schema_content = {"type": "object", "$defs": {"Item": {"type": "object"}}}

        original_meta = {"key": "value", "nested": {"data": 123}}
        schema_entity = JsonSchema(
            id=42, content=schema_content, is_synthetic=False, meta=original_meta
        )

        new_schemas = extract_subschemas_from_schema(schema_entity)

        assert len(new_schemas) == 1
        assert new_schemas[0].meta["original_schema_meta"] == original_meta


class TestSchemaValidation:
    def test_valid_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        assert validate_schema(schema) is True

    def test_invalid_schema_wrong_type(self):
        schema = {"type": "invalid_type"}

        assert validate_schema(schema) is False

    def test_invalid_schema_bad_structure(self):
        schema = {"type": "object", "properties": "not a dict"}

        assert validate_schema(schema) is False

    def test_valid_array_schema(self):
        schema = {"type": "array", "items": {"type": "string"}}

        assert validate_schema(schema) is True

    def test_complex_valid_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                }
            },
        }

        assert validate_schema(schema) is True


class TestUrlSchemaExtraction:
    @patch("any2json.data_engine.helpers.httpx.get")
    def test_fetch_schema_from_url_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        mock_get.return_value = mock_response

        result = fetch_schema_from_url("https://example.com/schema.json")

        assert result == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        mock_get.assert_called_once()

    @patch("any2json.data_engine.helpers.httpx.get")
    def test_fetch_schema_from_url_failure(self, mock_get):
        mock_get.side_effect = Exception("Network error")

        result = fetch_schema_from_url("https://example.com/schema.json")

        assert result is None

    def test_extract_schema_from_ref_internal(self):
        base_schema = {
            "type": "object",
            "$defs": {
                "User": {"type": "object", "properties": {"id": {"type": "integer"}}}
            },
        }

        schema, path = extract_schema_from_ref("#/$defs/User", base_schema)

        assert schema == {"type": "object", "properties": {"id": {"type": "integer"}}}
        assert path == "ref:#/$defs/User"

    def test_extract_schema_from_ref_internal_not_found(self):
        base_schema = {"type": "object", "$defs": {}}

        schema, path = extract_schema_from_ref("#/$defs/NonExistent", base_schema)

        assert schema is None
        assert path == "ref:#/$defs/NonExistent"

    @patch("any2json.data_engine.helpers.fetch_schema_from_url")
    def test_extract_schema_from_ref_url_simple(self, mock_fetch):
        mock_fetch.return_value = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        schema, path = extract_schema_from_ref("https://example.com/schema.json", None)

        assert schema == {"type": "object", "properties": {"name": {"type": "string"}}}
        assert path == "url:https://example.com/schema.json"

    @patch("any2json.data_engine.helpers.fetch_schema_from_url")
    def test_extract_schema_from_ref_url_with_pointer(self, mock_fetch):
        mock_fetch.return_value = {
            "type": "object",
            "$defs": {
                "culinarySpecialty": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        }

        schema, path = extract_schema_from_ref(
            "https://example.com/schema.json#/$defs/culinarySpecialty", None
        )

        assert schema == {"type": "object", "properties": {"name": {"type": "string"}}}
        assert path == "url:https://example.com/schema.json#/$defs/culinarySpecialty"

    @patch("any2json.data_engine.helpers.fetch_schema_from_url")
    def test_extract_schema_from_ref_url_failure(self, mock_fetch):
        mock_fetch.return_value = None

        schema, path = extract_schema_from_ref("https://example.com/schema.json", None)

        assert schema is None
        assert path == "url:https://example.com/schema.json"


class TestImplicitSubschemasWithRefs:
    def test_extract_ref_internal(self):
        schema = {
            "type": "object",
            "properties": {"person": {"$ref": "#/$defs/Person"}},
            "$defs": {
                "Person": {"type": "object", "properties": {"name": {"type": "string"}}}
            },
        }

        subschemas = extract_implicit_subschemas(schema)

        assert len(subschemas) >= 1
        paths = [path for _, path in subschemas]
        assert any("ref:#/$defs/Person" in path for path in paths)

    @patch("any2json.data_engine.helpers.fetch_schema_from_url")
    def test_extract_ref_url(self, mock_fetch):
        mock_fetch.return_value = {
            "type": "object",
            "properties": {"specialty": {"type": "string"}},
        }

        schema = {
            "type": "object",
            "properties": {
                "culinarySpecialties": {
                    "type": "array",
                    "items": {
                        "$ref": "https://example.com/schema.json#/$defs/culinarySpecialty"
                    },
                }
            },
        }

        subschemas = extract_implicit_subschemas(schema)

        assert len(subschemas) >= 1
        paths = [path for _, path in subschemas]
        assert any("culinarySpecialties" in path for path in paths)

    def test_extract_ref_nested_in_items(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Item"},
                }
            },
            "$defs": {
                "Item": {"type": "object", "properties": {"id": {"type": "integer"}}}
            },
        }

        subschemas = extract_implicit_subschemas(schema)

        assert len(subschemas) >= 1


class TestExpandRefsInSchema:
    def test_expand_simple_ref(self):
        base_schema = {
            "$defs": {
                "Person": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        }
        schema_with_ref = {"$ref": "#/$defs/Person"}

        expanded = expand_refs_in_schema(schema_with_ref, base_schema)

        assert expanded == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        assert "$ref" not in expanded

    def test_expand_nested_refs(self):
        base_schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}},
                },
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"$ref": "#/$defs/Address"},
                    },
                },
            }
        }
        schema_with_ref = {"$ref": "#/$defs/Person"}

        expanded = expand_refs_in_schema(schema_with_ref, base_schema)

        assert expanded["type"] == "object"
        assert "name" in expanded["properties"]
        assert "address" in expanded["properties"]
        assert expanded["properties"]["address"]["type"] == "object"
        assert "street" in expanded["properties"]["address"]["properties"]
        assert "$ref" not in str(expanded)

    def test_expand_refs_in_items(self):
        base_schema = {
            "$defs": {
                "Item": {"type": "object", "properties": {"id": {"type": "integer"}}}
            }
        }
        schema = {"type": "array", "items": {"$ref": "#/$defs/Item"}}

        expanded = expand_refs_in_schema(schema, base_schema)

        assert expanded["type"] == "array"
        assert expanded["items"]["type"] == "object"
        assert "id" in expanded["items"]["properties"]
        assert "$ref" not in str(expanded)

    def test_expand_refs_preserves_other_keys(self):
        base_schema = {
            "$defs": {
                "Item": {"type": "object", "properties": {"id": {"type": "integer"}}}
            }
        }
        schema = {"$ref": "#/$defs/Item", "description": "An item"}

        expanded = expand_refs_in_schema(schema, base_schema)

        assert expanded["type"] == "object"
        assert "id" in expanded["properties"]
        assert expanded.get("description") == "An item"


class TestExtractSubschemasWithValidation:
    def test_extract_only_valid_schemas(self):
        schema_content = {
            "type": "object",
            "properties": {
                "valid": {"type": "object", "properties": {"name": {"type": "string"}}},
                "invalid": {"type": "invalid_type"},
            },
        }

        schema_entity = JsonSchema(
            id=1, content=schema_content, is_synthetic=False, meta={"source": "test"}
        )

        new_schemas = extract_subschemas_from_schema(schema_entity)

        assert len(new_schemas) >= 1
        for schema in new_schemas:
            assert validate_schema(schema.content)


class TestRealWorldComplexSchema:
    def test_extract_all_subschemas_from_brand_recognition_schema(self):
        schema_content = {
            "$defs": {
                "brandRecognitionStrategy": {
                    "properties": {
                        "methodology": {
                            "items": {"$ref": "#/$defs/recognitionMethod"},
                            "type": ["array", "null"],
                        },
                        "name": {"type": ["string", "null"]},
                    },
                    "type": ["object", "null"],
                },
                "productPlacementTechnique": {
                    "properties": {
                        "description": {"type": ["string", "null"]},
                        "name": {"type": ["string", "null"]},
                    },
                    "type": ["object", "null"],
                },
                "recognitionMethod": {
                    "properties": {
                        "effectiveness": {"type": ["number", "null"]},
                        "technique": {"type": ["string", "null"]},
                    },
                    "type": ["object", "null"],
                },
            },
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {
                "advertisingTheory": {"type": ["string", "null"]},
                "brandRecognitionStrategies": {
                    "items": {"$ref": "#/$defs/brandRecognitionStrategy"},
                    "type": ["array", "null"],
                },
                "productPlacementTechniques": {
                    "items": {"$ref": "#/$defs/productPlacementTechnique"},
                    "type": ["array", "null"],
                },
            },
            "type": ["object", "null"],
        }

        schema_entity = JsonSchema(
            id=100,
            content=schema_content,
            is_synthetic=False,
            meta={"source": "real_world_test"},
        )

        new_schemas = extract_subschemas_from_schema(schema_entity)

        expected_subschemas = [
            {
                "content": {
                    "properties": {
                        "methodology": {
                            "items": {
                                "properties": {
                                    "effectiveness": {"type": ["number", "null"]},
                                    "technique": {"type": ["string", "null"]},
                                },
                                "type": ["object", "null"],
                            },
                            "type": ["array", "null"],
                        },
                        "name": {"type": ["string", "null"]},
                    },
                    "type": ["object", "null"],
                },
                "path": "$defs.brandRecognitionStrategy",
            },
            {
                "content": {
                    "properties": {
                        "description": {"type": ["string", "null"]},
                        "name": {"type": ["string", "null"]},
                    },
                    "type": ["object", "null"],
                },
                "path": "$defs.productPlacementTechnique",
            },
            {
                "content": {
                    "properties": {
                        "effectiveness": {"type": ["number", "null"]},
                        "technique": {"type": ["string", "null"]},
                    },
                    "type": ["object", "null"],
                },
                "path": "$defs.recognitionMethod",
            },
            {
                "content": {
                    "items": {
                        "properties": {
                            "methodology": {
                                "items": {
                                    "properties": {
                                        "effectiveness": {"type": ["number", "null"]},
                                        "technique": {"type": ["string", "null"]},
                                    },
                                    "type": ["object", "null"],
                                },
                                "type": ["array", "null"],
                            },
                            "name": {"type": ["string", "null"]},
                        },
                        "type": ["object", "null"],
                    },
                    "type": ["array", "null"],
                },
                "path": "properties.brandRecognitionStrategies",
            },
            {
                "content": {
                    "items": {
                        "properties": {
                            "description": {"type": ["string", "null"]},
                            "name": {"type": ["string", "null"]},
                        },
                        "type": ["object", "null"],
                    },
                    "type": ["array", "null"],
                },
                "path": "properties.productPlacementTechniques",
            },
            {
                "content": {
                    "properties": {
                        "methodology": {
                            "items": {
                                "properties": {
                                    "effectiveness": {"type": ["number", "null"]},
                                    "technique": {"type": ["string", "null"]},
                                },
                                "type": ["object", "null"],
                            },
                            "type": ["array", "null"],
                        },
                        "name": {"type": ["string", "null"]},
                    },
                    "type": ["object", "null"],
                },
                "path": "properties.brandRecognitionStrategies.items.ref:#/$defs/brandRecognitionStrategy",
            },
            {
                "content": {
                    "items": {
                        "properties": {
                            "effectiveness": {"type": ["number", "null"]},
                            "technique": {"type": ["string", "null"]},
                        },
                        "type": ["object", "null"],
                    },
                    "type": ["array", "null"],
                },
                "path": "properties.brandRecognitionStrategies.items.ref:#/$defs/brandRecognitionStrategy.properties.methodology",
            },
            {
                "content": {
                    "properties": {
                        "effectiveness": {"type": ["number", "null"]},
                        "technique": {"type": ["string", "null"]},
                    },
                    "type": ["object", "null"],
                },
                "path": "properties.brandRecognitionStrategies.items.ref:#/$defs/brandRecognitionStrategy.properties.methodology.items.ref:#/$defs/recognitionMethod",
            },
            {
                "content": {
                    "properties": {
                        "description": {"type": ["string", "null"]},
                        "name": {"type": ["string", "null"]},
                    },
                    "type": ["object", "null"],
                },
                "path": "properties.productPlacementTechniques.items.ref:#/$defs/productPlacementTechnique",
            },
        ]

        assert len(new_schemas) == len(expected_subschemas)

        extracted_schemas_by_path = {
            schema.meta["extraction_path"]: schema.content for schema in new_schemas
        }

        for expected in expected_subschemas:
            assert expected["path"] in extracted_schemas_by_path
            assert extracted_schemas_by_path[expected["path"]] == expected["content"]

        for schema in new_schemas:
            assert schema.parent_schema_id == 100
            assert schema.meta["source"] == "extracted_subschema"
            assert schema.meta["original_schema_id"] == 100
            assert "extraction_path" in schema.meta
            assert validate_schema(schema.content)
