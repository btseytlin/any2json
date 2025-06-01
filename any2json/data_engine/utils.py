import json
from any2json.containers import InputJSONChunk
import random
from datasets import Dataset


def get_chunks_from_record(
    record: dict | list,
    max_depth: int = 3,
    **kwargs,
) -> list[InputJSONChunk]:
    """
    Get all the chunks from an arbitrary nested dictionary that might contain lists and dictionaries.

    Will recursively traverse the dictionary, pick out values that are not primitive types and return a list of InputJSONChunk objects.

    Args:
        record: A dictionary that might contain lists and dictionaries.

    Returns:
        A list of InputJSONChunk objects.
    """
    chunks = []

    if isinstance(record, (list, dict)):
        if isinstance(record, dict):
            any_value = False
            for k, v in record.items():
                if v:
                    any_value = True
                    break

        if isinstance(record, list):
            for v in record:
                if v:
                    any_value = True
                    break

        if not any_value:
            return chunks

        chunks.append(InputJSONChunk(data=record, id=None, **kwargs))

    if max_depth == 0:
        return chunks

    if isinstance(record, dict):
        for k, v in record.items():
            if isinstance(v, (list, dict)):
                chunks.extend(get_chunks_from_record(v, max_depth - 1, **kwargs))
    elif isinstance(record, list):
        for v in record:
            chunks.extend(get_chunks_from_record(v, max_depth - 1, **kwargs))

    return chunks


def deduplicate_chunks(chunks: list[InputJSONChunk]) -> list[InputJSONChunk]:
    seen_hashes = set()

    deduped_chunks = []
    for chunk in chunks:
        hash_value = hash(json.dumps(chunk.data))
        if hash_value not in seen_hashes:
            seen_hashes.add(hash_value)
            deduped_chunks.append(chunk)

    return deduped_chunks


def generate_chunks_from_json_dataset(
    dataset: Dataset,
    dataset_name: str,
    num_chunks_per_dataset: int,
) -> list[InputJSONChunk]:
    chunks = []

    for i, record in enumerate(dataset):
        chunks.extend(
            get_chunks_from_record(
                record,
                source_dataset_name=dataset_name,
                source_dataset_index=i,
            )
        )

    chunks = deduplicate_chunks(chunks)
    chunks = random.sample(chunks, min(num_chunks_per_dataset, len(chunks)))

    return chunks
