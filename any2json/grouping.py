from collections import Counter
import copy
import json
import hashlib
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from any2json.database.models import Chunk, SchemaConversion
from any2json.enums import ContentType
from any2json.utils import logger, stringify_content
from sqlalchemy.orm import Session
from tqdm.auto import tqdm


def stringify_for_hash(value: object, format: ContentType | str) -> str:
    if not isinstance(value, str):
        value = stringify_content(value, format)
    signature = "".join(sorted(value.lower()))
    return signature


def digest_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dataset_signature_from_conversion(
    schema_conversion: SchemaConversion,
) -> str | None:
    input_doc = (
        schema_conversion.input_chunk.parent_document
        if schema_conversion.input_chunk
        else None
    )
    output_doc = (
        schema_conversion.output_chunk.parent_document
        if schema_conversion.output_chunk
        else None
    )
    if input_doc and input_doc.meta and "source_dataset_index" in input_doc.meta:
        idx = input_doc.meta["source_dataset_index"]
        sig = f"{input_doc.source}:{idx}"
        return sig
    if output_doc and output_doc.meta and "source_dataset_index" in output_doc.meta:
        idx = output_doc.meta["source_dataset_index"]
        sig = f"{output_doc.source}:{idx}"
        return sig
    schema_meta = schema_conversion.schema.meta if schema_conversion.schema else None
    if schema_meta and "source_dataset_index" in schema_meta:
        ds = schema_meta.get("source_dataset") or ""
        idx = schema_meta["source_dataset_index"]
        sig = f"{ds}:{idx}"
        return sig
    return None


def build_signatures(schema_conversion: SchemaConversion) -> set[str]:
    sigs: set[str] = set()
    if schema_conversion.input_chunk:
        s = stringify_for_hash(
            schema_conversion.input_chunk.content,
            schema_conversion.input_chunk.content_type,
        )
        h = digest_string(s)
        sigs.add(f"input:{h}")
    if schema_conversion.schema:
        s = stringify_for_hash(
            schema_conversion.schema.content,
            ContentType.JSON,
        )
        h = digest_string(s)
        sigs.add(f"schema:{h}")
    if schema_conversion.output_chunk:
        s = stringify_for_hash(
            schema_conversion.output_chunk.content,
            schema_conversion.output_chunk.content_type,
        )
        h = digest_string(s)
        sigs.add(f"output:{h}")
    ds = dataset_signature_from_conversion(schema_conversion)
    if ds is not None:
        sigs.add(f"dataset:{ds}")
    return sigs


def find_parent_id(parents: dict[int, int], i: int) -> int:
    if parents[i] != i:
        parents[i] = find_parent_id(parents, parents[i])
    return parents[i]


def union_ids(parents: dict[int, int], a: int, b: int) -> None:
    ra = find_parent_id(parents, a)
    rb = find_parent_id(parents, b)
    if ra == rb:
        return
    if ra < rb:
        parents[rb] = ra
    else:
        parents[ra] = rb


def connected_component_groups(signature_map: dict[int, set[str]]) -> dict[int, int]:
    ids = list(signature_map.keys())
    parents: dict[int, int] = {i: i for i in ids}
    seen: dict[str, int] = {}
    for i in tqdm(ids, desc="Processing entities"):
        for sig in signature_map[i]:
            if sig in seen:
                union_ids(parents, i, seen[sig])
            else:
                seen[sig] = i
    roots: dict[int, int] = {}
    groups: dict[int, int] = {}
    next_group = 0
    for i in tqdm(ids, desc="Assigning groups"):
        r = find_parent_id(parents, i)
        if r not in roots:
            roots[r] = next_group
            next_group += 1
        groups[i] = roots[r]
    return groups


def group_conversions(
    conversions: list[SchemaConversion],
) -> dict[int, int] | None:
    logger.info(f"Building signatures for {len(conversions)} conversions")
    signature_map = {int(c.id): build_signatures(c) for c in tqdm(conversions)}
    logger.info(f"Finding connected components")
    groups = connected_component_groups(signature_map)
    return groups


def assign_groups(
    db_session: Session,
):
    query = (
        select(SchemaConversion)
        .options(
            joinedload(SchemaConversion.input_chunk).joinedload(Chunk.parent_document),
            joinedload(SchemaConversion.output_chunk).joinedload(Chunk.parent_document),
            joinedload(SchemaConversion.schema),
        )
        .order_by(SchemaConversion.id)
    )
    conversions = db_session.execute(query).scalars().all()
    logger.info(f"Loaded {len(conversions)} schema conversions")
    conversion_to_group_map = group_conversions(conversions)
    logger.info(
        f"Computed {len(set(conversion_to_group_map.values()))} groups from connected components"
    )
    per_group_size = Counter(conversion_to_group_map.values())
    logger.info(f"10 largest groups: {per_group_size.most_common(10)}")
    size_distribution = Counter(per_group_size.values())
    logger.info(f"20 most common group sizes: {size_distribution.most_common(20)}")

    # Rename group such that the largest group is 0, the second largest is 1, etc.
    group_rename_map = {}
    for rank, (group_id, _) in enumerate(
        sorted(per_group_size.items(), key=lambda x: x[1], reverse=True)
    ):
        group_rename_map[group_id] = rank
    logger.info(f"Created group rename map")

    for schema_conversion in conversions:
        group_id = conversion_to_group_map[schema_conversion.id]
        group_id = group_rename_map[group_id]
        logger.debug(f"Assigning {schema_conversion.id} to group {group_id}")
        new_meta = copy.deepcopy(schema_conversion.meta or {})
        new_meta["group"] = group_id
        schema_conversion.meta = new_meta
        db_session.add(schema_conversion)
