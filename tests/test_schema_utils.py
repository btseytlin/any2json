import fastjsonschema
from any2json.schema_utils import to_supported_json_schema


class TestSupportedSchema:
    def test_all_fields_become_nullable(self):
        schema = {
            "type": "object",
            "properties": {
                "flights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "departureTime": {"type": "date"},
                        },
                    },
                },
                "queryTime": {"type": "date"},
            },
        }

        expected = {
            "type": ["object", "null"],
            "properties": {
                "flights": {
                    "type": ["array", "null"],
                    "items": {
                        "type": ["object", "null"],
                        "properties": {
                            "departureTime": {"type": ["string", "null"]},
                        },
                    },
                },
                "queryTime": {"type": ["string", "null"]},
            },
        }

        assert to_supported_json_schema(schema) == expected
        assert fastjsonschema.compile(expected)

    def test_converts_date_types(self):
        schema = {
            "type": "object",
            "properties": {
                "flights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "departureTime": {"type": "date"},
                        },
                    },
                },
                "queryTime": {"type": "date"},
            },
        }

        expected = {
            "type": ["object", "null"],
            "properties": {
                "flights": {
                    "type": ["array", "null"],
                    "items": {
                        "type": ["object", "null"],
                        "properties": {
                            "departureTime": {"type": ["string", "null"]},
                        },
                    },
                },
                "queryTime": {"type": ["string", "null"]},
            },
        }

        assert to_supported_json_schema(schema) == expected

    def test_to_supported_json_schema_real_examples(self):
        schema = {
            "type": "object",
            "properties": {
                "flights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "flightNumber": {"type": "string"},
                            "departureAirport": {"type": "string"},
                            "arrivalCity": {"type": "string"},
                            "departureTime": {"type": "string", "format": "date-time"},
                            "arrivalTime": {"type": "string", "format": "date-time"},
                            "price": {
                                "type": "object",
                                "properties": {
                                    "currency": {"type": "string"},
                                    "amount": {"type": "number"},
                                },
                                "required": ["currency", "amount"],
                            },
                        },
                        "required": [
                            "flightNumber",
                            "departureAirport",
                            "arrivalCity",
                            "departureTime",
                            "arrivalTime",
                            "price",
                        ],
                    },
                    "minItems": 5,
                    "maxItems": 20,
                }
            },
            "required": ["flights"],
        }

        expected = {
            "type": ["object", "null"],
            "properties": {
                "flights": {
                    "type": ["array", "null"],
                    "items": {
                        "type": ["object", "null"],
                        "properties": {
                            "flightNumber": {"type": ["string", "null"]},
                            "departureAirport": {"type": ["string", "null"]},
                            "arrivalCity": {"type": ["string", "null"]},
                            "departureTime": {"type": ["string", "null"]},
                            "arrivalTime": {"type": ["string", "null"]},
                            "price": {
                                "type": ["object", "null"],
                                "properties": {
                                    "currency": {"type": ["string", "null"]},
                                    "amount": {
                                        "type": ["number", "null"],
                                    },
                                },
                            },
                        },
                    },
                }
            },
        }

        assert to_supported_json_schema(schema) == expected
        assert fastjsonschema.compile(expected)

        schema = {
            "type": "object",
            "properties": {
                "tokamak": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "location": {"type": "string"},
                        "operation_schedule": {
                            "type": "object",
                            "properties": {
                                "start_date": {"type": "date"},
                                "end_date": {"type": "date"},
                            },
                            "required": ["start_date", "end_date"],
                        },
                        "key_features": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "feature_name": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["feature_name", "description"],
                            },
                        },
                        "research_findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "finding_title": {"type": "string"},
                                    "abstract": {"type": "string"},
                                },
                                "required": ["finding_title", "abstract"],
                            },
                        },
                    },
                    "required": [
                        "name",
                        "location",
                        "operation_schedule",
                        "key_features",
                        "research_findings",
                    ],
                }
            },
            "required": ["tokamak"],
        }

        expected = {
            "type": ["object", "null"],
            "properties": {
                "tokamak": {
                    "type": ["object", "null"],
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "location": {"type": ["string", "null"]},
                        "operation_schedule": {
                            "type": ["object", "null"],
                            "properties": {
                                "start_date": {"type": ["string", "null"]},
                                "end_date": {"type": ["string", "null"]},
                            },
                        },
                        "key_features": {
                            "type": ["array", "null"],
                            "items": {
                                "type": ["object", "null"],
                                "properties": {
                                    "feature_name": {"type": ["string", "null"]},
                                    "description": {"type": ["string", "null"]},
                                },
                            },
                        },
                        "research_findings": {
                            "type": ["array", "null"],
                            "items": {
                                "type": ["object", "null"],
                                "properties": {
                                    "finding_title": {"type": ["string", "null"]},
                                    "abstract": {"type": ["string", "null"]},
                                },
                            },
                        },
                    },
                }
            },
        }

    def test_refs_minimal(self):
        schema = {
            "$id": "https://example.com/spanish-flu",
            "type": "object",
            "properties": {
                "cities": {"$ref": "#/$defs/CityData"},
            },
            "additionalProperties": False,
            "$defs": {
                "CityData": {
                    "$id": "https://example.com/city-data",
                    "type": "object",
                    "properties": {
                        "city_name": {"type": "string"},
                        "mortality_rate": {"type": "integer"},
                        "population_size": {"type": "integer"},
                        # "government_response": {"$ref": "#/$defs/GovernmentResponse"},
                    },
                    "required": ["city_name"],
                    "additionalProperties": False,
                },
            },
        }

        expected = {
            "$id": "https://example.com/spanish-flu",
            "type": ["object", "null"],
            "properties": {
                "cities": {"$ref": "#/$defs/CityData"},
            },
            "$defs": {
                "CityData": {
                    "$id": "https://example.com/city-data",
                    "type": ["object", "null"],
                    "properties": {
                        "city_name": {"type": ["string", "null"]},
                        "mortality_rate": {"type": ["integer", "null"]},
                        "population_size": {"type": ["integer", "null"]},
                    },
                },
            },
        }

        assert to_supported_json_schema(schema) == expected
        assert fastjsonschema.compile(expected)
