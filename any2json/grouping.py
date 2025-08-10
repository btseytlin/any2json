from collections import Counter
import random
import copy
import json
import hashlib
from typing import Any
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


def train_test_split_groups(
    items: list[Any],
    groups: list[Any],
    test_size: int,
    random_state: int,
    exclude_top_k_groups: int = 2,
    selection_slack: int = 50,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """A function to assemble a train/test split of items where items are grouped in uneven sizes.

    Args:
        items: list of items to split
        groups: list of groups for each item
        test_size: desired approximate size of the test set
        random_state: random state for the split

    Returns:
        train_items: list of items in the train set
        test_items: list of items in the test set
        train_groups: list of groups in the train set
        test_groups: list of groups in the test set
    """
    assert len(items) == len(groups) and 0 < test_size < len(items)
    group_to_idx: dict[Any, list[int]] = {}
    for i, g in enumerate(groups):
        group_to_idx.setdefault(g, []).append(i)
    rng = random.Random(random_state)
    sizes = {g: len(v) for g, v in group_to_idx.items()}
    sorted_groups = sorted(sizes, key=sizes.get, reverse=True)
    k = max(0, min(exclude_top_k_groups, len(sorted_groups)))
    excluded = set(sorted_groups[:k])
    order = [g for g in group_to_idx.keys() if g not in excluded]
    rng.shuffle(order)
    target = test_size
    test_idx: set[int] = set()
    test_count = 0
    logger.info(f"Target test size: {target}")
    for _, g in enumerate(order):
        size = len(group_to_idx[g])
        logger.debug(f"Group {g} size: {size}")
        current_dist = abs(target - test_count)
        logger.debug(f"Current test size: {test_count}")
        logger.debug(f"Current distance: {current_dist}")

        if size > current_dist:
            logger.debug(f"Skipping group {g} because it is too large")
            continue

        new_count = test_count + size
        candidate_dist = abs(target - new_count)
        if candidate_dist <= current_dist + max(0, selection_slack):
            if rng.random() < 0.5:
                logger.debug(f"Skipping group {g} randomly")
                continue
            logger.debug(f"Adding group {g} to test set")
            test_idx.update(group_to_idx[g])
            test_count = new_count

    train_idx = [i for i in range(len(items)) if i not in test_idx]
    test_idx_list = sorted(test_idx)
    train_items = [items[i] for i in train_idx]
    test_items = [items[i] for i in test_idx_list]
    train_groups = [groups[i] for i in train_idx]
    test_groups = [groups[i] for i in test_idx_list]

    train_group_sizes = Counter(train_groups)
    logger.info(f"Train group sizes (top 10): {train_group_sizes.most_common(10)}")
    test_group_sizes = Counter(test_groups)
    logger.info(f"Test group sizes (top 10): {test_group_sizes.most_common(10)}")

    return train_items, test_items, train_groups, test_groups
