from copy import deepcopy


def remove_list_types_from_schema(schema: dict) -> dict:
    new_schema = deepcopy(schema)
    for prop in new_schema["properties"].values():
        if (
            isinstance(prop["type"], list)
            and len(prop["type"]) == 2
            and prop["type"][1] == "null"
        ):
            prop["type"] = prop["type"][0]

        if prop["type"] == "object":
            prop["type"] = remove_list_types_from_schema(prop)

    return new_schema
