from copy import deepcopy
import logging

logger = logging.getLogger("any2json")


def to_supported_type(type_: str | dict | list) -> str:
    type_replaces = {
        "date": "string",
    }
    if isinstance(type_, str):
        if type_ == "null":
            return "null"
        else:
            return [type_replaces.get(type_, type_), "null"]

    if isinstance(type_, list):
        type_ = [type_replaces.get(item, item) for item in type_]
        if "null" not in type_:
            type_.append("null")
        return type_

    type_dict = deepcopy(type_)
    type_replaces = {
        "date": "string",
    }
    drop_type_keys = ["enum", "format"]

    type_dict = {k_: v_ for k_, v_ in type_dict.items() if k_ not in drop_type_keys}
    type_dict["type"] = to_supported_type(type_dict["type"])
    return type_dict


def to_supported_json_schema(
    schema: dict | list,
) -> dict | list:
    schema = deepcopy(schema)

    error_on_keys = ["allOf", "oneOf", "anyOf", "not"]

    drop_keys = [
        "extends",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "minLength",
        "maxLength",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "enum",
        "format",
        "pattern",
        "patternProperties",
        "default",
        "description",
        "title",
        "$comment",
        "uniqueItems",
    ]

    if isinstance(schema, list):
        return [to_supported_json_schema(item) for item in schema]

    if isinstance(schema, dict):
        schema = {k: v for k, v in schema.items() if k not in drop_keys}

        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
            schema["additionalProperties"] = False

        for k, v in schema.items():
            if k in error_on_keys:
                raise ValueError(f"Schema contains {k} key, which is not supported")

        if "definitions" in schema:
            schema["$defs"] = schema["definitions"]
            del schema["definitions"]

        for k, v in schema.items():
            if k == "type":
                schema[k] = to_supported_type(v)
            elif k == "properties":
                schema[k] = {k_: to_supported_json_schema(v_) for k_, v_ in v.items()}
            elif k == "items":
                schema[k] = to_supported_json_schema(v)
            elif k == "$defs":
                schema["$defs"] = {
                    k_: to_supported_json_schema(v_) for k_, v_ in v.items()
                }
            elif isinstance(v, (dict, list)):
                schema[k] = to_supported_json_schema(v)
    return schema
