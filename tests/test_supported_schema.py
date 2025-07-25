import fastjsonschema
from any2json.utils import to_supported_json_schema


class TestSupportedSchema:

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
            "type": "object",
            "properties": {
                "flights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "departureTime": {"type": "string"},
                        },
                    },
                },
                "queryTime": {"type": "string"},
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
                            "departureTime": {"type": "string"},
                            "arrivalTime": {"type": "string"},
                            "price": {
                                "type": "object",
                                "properties": {
                                    "currency": {"type": "string"},
                                    "amount": {"type": "number"},
                                },
                            },
                        },
                    },
                }
            },
        }

        assert to_supported_json_schema(schema) == expected

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
                                "start_date": {"type": "string"},
                                "end_date": {"type": "string"},
                            },
                        },
                        "key_features": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "feature_name": {"type": "string"},
                                    "description": {"type": "string"},
                                },
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
            "type": "object",
            "properties": {
                "cities": {"$ref": "#/$defs/CityData"},
            },
            "$defs": {
                "CityData": {
                    "$id": "https://example.com/city-data",
                    "type": "object",
                    "properties": {
                        "city_name": {"type": "string"},
                        "mortality_rate": {"type": "integer"},
                        "population_size": {"type": "integer"},
                    },
                },
            },
        }

        assert to_supported_json_schema(schema) == expected
        assert fastjsonschema.compile(expected)
