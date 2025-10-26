from copy import deepcopy
import json
import logging

import fastjsonschema
import httpx
from tqdm import tqdm

from any2json.database.models import JsonSchema


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
    expand_refs: bool = True,
) -> dict | list:
    schema = deepcopy(schema)

    if expand_refs:
        schema = expand_refs_in_schema(schema, schema)
    expand_refs = False

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
        return [
            to_supported_json_schema(item, expand_refs=expand_refs) for item in schema
        ]

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
                schema[k] = {
                    k_: to_supported_json_schema(v_, expand_refs=expand_refs)
                    for k_, v_ in v.items()
                }
            elif k == "items":
                schema[k] = to_supported_json_schema(v, expand_refs=expand_refs)
            elif k == "$defs":
                schema["$defs"] = {
                    k_: to_supported_json_schema(v_, expand_refs=expand_refs)
                    for k_, v_ in v.items()
                }
            elif isinstance(v, (dict, list)):
                schema[k] = to_supported_json_schema(v, expand_refs=expand_refs)
    return schema


def fetch_schema_from_url(url: str, timeout: int = 10) -> dict:
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise ValueError(f"Failed to fetch schema from {url}: {e}") from e


def extract_schema_from_ref(
    ref_value: str, base_schema: dict | None = None
) -> tuple[dict, str]:
    if ref_value == "#":
        if not base_schema:
            raise ValueError(
                f"Cannot resolve reference {ref_value} without base schema"
            )
        return base_schema, "ref:#"
    elif ref_value.startswith("#/"):
        if not base_schema:
            raise ValueError(
                f"Cannot resolve reference {ref_value} without base schema"
            )
        pointer = ref_value[2:]
        parts = pointer.split("/")
        current = base_schema
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ValueError(
                    f"Cannot resolve reference {ref_value} in base schema - path not found at '{part}'"
                )
        if not isinstance(current, dict):
            raise ValueError(f"Reference {ref_value} does not point to a schema object")
        return current, f"ref:{ref_value}"
    elif "#/" in ref_value and base_schema:
        schema_id_part, pointer = ref_value.split("#/", 1)
        base_schema_id = base_schema.get("$id", "")
        if base_schema_id and (
            schema_id_part == base_schema_id
            or schema_id_part.endswith(base_schema_id)
            or base_schema_id.endswith(schema_id_part)
        ):
            parts = pointer.split("/")
            current = base_schema
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    raise ValueError(
                        f"Cannot resolve reference {ref_value} in base schema with $id={base_schema_id} - path not found at '{part}'"
                    )
            if not isinstance(current, dict):
                raise ValueError(
                    f"Reference {ref_value} does not point to a schema object"
                )
            return current, f"ref:{ref_value}"

    if ref_value.startswith("http://") or ref_value.startswith("https://"):
        if "#/" in ref_value:
            url, json_pointer = ref_value.split("#/", 1)
            schema = fetch_schema_from_url(url)
            parts = json_pointer.split("/")
            current = schema
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    raise ValueError(
                        f"Cannot resolve JSON pointer {json_pointer} in schema from {url}"
                    )
            if not isinstance(current, dict):
                raise ValueError(
                    f"Reference {ref_value} does not point to a schema object"
                )
            return current, f"url:{url}#/{json_pointer}"
        else:
            schema = fetch_schema_from_url(ref_value)
            if not isinstance(schema, dict):
                raise ValueError(
                    f"Schema fetched from {ref_value} is not a valid object"
                )
            return schema, f"url:{ref_value}"
    else:
        raise ValueError(f"Unsupported reference format: {ref_value}")


def extract_explicit_subschemas(
    schema: dict,
) -> list[tuple[dict, str]]:
    subschemas = []

    if "$defs" in schema:
        for def_key, def_schema in schema["$defs"].items():
            if isinstance(def_schema, dict):
                subschemas.append((def_schema, f"$defs.{def_key}"))

    if "definitions" in schema:
        for def_key, def_schema in schema["definitions"].items():
            if isinstance(def_schema, dict):
                subschemas.append((def_schema, f"definitions.{def_key}"))

    return subschemas


def is_complex_schema(schema: dict) -> bool:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return "object" in schema_type or "array" in schema_type
    return schema_type in ["object", "array"]


def extract_implicit_subschemas(
    schema: dict,
    path: str = "",
    base_schema: dict | None = None,
) -> list[tuple[dict, str]]:
    subschemas = []

    if base_schema is None:
        base_schema = schema

    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_value = schema["$ref"]
            try:
                ref_schema, ref_path = extract_schema_from_ref(ref_value, base_schema)
                current_path = f"{path}.{ref_path}" if path else ref_path
                if is_complex_schema(ref_schema):
                    subschemas.append((ref_schema, current_path))
                subschemas.extend(
                    extract_implicit_subschemas(ref_schema, current_path, base_schema)
                )
            except ValueError as e:
                raise ValueError(
                    f"Failed to extract schema from $ref '{ref_value}' at path '{path}': {e}"
                ) from e

        if "properties" in schema:
            for prop_key, prop_schema in schema["properties"].items():
                if isinstance(prop_schema, dict):
                    current_path = (
                        f"{path}.properties.{prop_key}"
                        if path
                        else f"properties.{prop_key}"
                    )
                    if is_complex_schema(prop_schema):
                        subschemas.append((prop_schema, current_path))
                    subschemas.extend(
                        extract_implicit_subschemas(
                            prop_schema, current_path, base_schema
                        )
                    )

        if "items" in schema:
            items_schema = schema["items"]
            if isinstance(items_schema, dict):
                current_path = f"{path}.items" if path else "items"
                if is_complex_schema(items_schema):
                    subschemas.append((items_schema, current_path))
                subschemas.extend(
                    extract_implicit_subschemas(items_schema, current_path, base_schema)
                )

        if "additionalProperties" in schema:
            add_props = schema["additionalProperties"]
            if isinstance(add_props, dict):
                current_path = (
                    f"{path}.additionalProperties" if path else "additionalProperties"
                )
                if is_complex_schema(add_props):
                    subschemas.append((add_props, current_path))
                subschemas.extend(
                    extract_implicit_subschemas(add_props, current_path, base_schema)
                )

    return subschemas


def expand_refs_in_schema(
    schema: dict, base_schema: dict, visited_refs: set[str] | None = None
) -> dict:
    if visited_refs is None:
        visited_refs = set()

    if "$ref" in schema:
        ref_value = schema["$ref"]

        if ref_value in visited_refs:
            raise ValueError(f"Recursive reference detected: {ref_value}")

        visited_refs.add(ref_value)

        try:
            ref_schema, _ = extract_schema_from_ref(ref_value, base_schema)
            expanded = expand_refs_in_schema(ref_schema, base_schema, visited_refs)
            other_keys = {k: v for k, v in schema.items() if k != "$ref"}
            if other_keys:
                expanded = {**expanded, **other_keys}

            visited_refs.discard(ref_value)
            return expanded
        except ValueError as e:
            raise ValueError(
                f"Failed to expand $ref '{ref_value}' in schema {base_schema}: {e}"
            ) from e

    expanded = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            expanded[key] = expand_refs_in_schema(value, base_schema, visited_refs)
        elif isinstance(value, list):
            expanded[key] = [
                (
                    expand_refs_in_schema(item, base_schema, visited_refs)
                    if isinstance(item, dict)
                    else item
                )
                for item in value
            ]
        else:
            expanded[key] = value

    if "$defs" in expanded:
        del expanded["$defs"]
    if "definitions" in expanded:
        del expanded["definitions"]

    return expanded


def validate_schema(schema: dict) -> bool:
    try:
        fastjsonschema.compile(schema)
        return True
    except Exception as e:
        logger.debug(f"Schema validation failed: {e}")
        return False


def extract_subschemas_from_schema(
    schema: JsonSchema,
) -> list[JsonSchema]:
    schema_content = (
        schema.content
        if isinstance(schema.content, dict)
        else json.loads(schema.content)
    )

    explicit_subschemas = extract_explicit_subschemas(schema_content)
    implicit_subschemas = extract_implicit_subschemas(schema_content)

    all_subschemas = explicit_subschemas + implicit_subschemas

    new_schemas = []
    for subschema_content, path in all_subschemas:
        expanded_content = expand_refs_in_schema(subschema_content, schema_content)

        if "$defs" in expanded_content:
            del expanded_content["$defs"]
        if "definitions" in expanded_content:
            del expanded_content["definitions"]

        if not validate_schema(expanded_content):
            logger.debug(
                f"Skipping invalid subschema at path {path} from schema {schema.id}"
            )
            continue

        meta = {
            "source": "extracted_subschema",
            "original_schema_id": schema.id,
            "extraction_path": path,
        }
        if schema.meta:
            meta["original_schema_meta"] = schema.meta

        new_schema = JsonSchema(
            content=expanded_content,
            is_synthetic=False,
            parent_schema_id=schema.id,
            meta=meta,
        )
        new_schemas.append(new_schema)

    return new_schemas
