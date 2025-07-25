import logging
import os
import time
import click
import json
from dotenv import load_dotenv
import httpx
import re
from tqdm import tqdm

from any2json.data_engine.helpers import deduplicate_chunks
from any2json.database.client import get_db_session
from any2json.database.models import Chunk, JsonSchema, SchemaConversion, SourceDocument
from any2json.enums import ContentType
from any2json.utils import (
    logger,
    configure_loggers,
    post_with_retry,
    extract_json_from_markdown,
)


def find_documents(
    client: httpx.Client,
    url: str,
    index: str,
    query: str,
) -> dict:
    payload = {"index": index, "query_type": "find", "query": query}
    try:
        response = post_with_retry(client, url, payload, timeout=30.0)
        response.raise_for_status()
        results = response.json()
        if error := results.get("error"):
            logger.error(f"Infinigram API error: {error}")
            return {}
        return results
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        logger.error(f"Infinigram API request failed: {e}")
        raise e


def get_document_by_rank(
    client: httpx.Client, url: str, index: str, shard_index: int, rank: int
) -> dict:
    payload = {
        "index": index,
        "query_type": "get_doc_by_rank",
        "query": "```json",
        "rank": rank,
        "max_disp_len": 10000,
        "s": shard_index,
    }
    try:
        logger.debug(f"Querying Infinigram API: POST {url} with payload {payload}")
        response = post_with_retry(client, url, payload, timeout=30.0)

        # logger.debug(f"{response.text=}")
        response.raise_for_status()
        doc_data = response.json()
        if error := doc_data.get("error"):
            logger.warning(f"Infinigram API error: {response.json()}")
            return {}
        return doc_data
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.warning(f"Failed to get doc rank {rank}: {e}")
        return {}


def process_document(doc_data: dict, rank: int) -> dict | None:
    document_content = "\n".join(span[0] for span in doc_data["spans"])

    logger.debug(f"Extracting JSON chunks from document:\n{document_content}")
    json_chunks = extract_json_from_markdown(document_content)

    logger.debug(f"Extracted {len(json_chunks)} JSON chunks")

    if not json_chunks:
        return None

    document_meta = {
        "doc_ix": doc_data.get("doc_ix"),
        "doc_len": doc_data.get("doc_len"),
        "disp_len": doc_data.get("disp_len"),
        "rank": rank,
    }
    return {
        "document": document_content,
        "json_chunks": json_chunks,
        "document_meta": document_meta,
    }


def fetch_and_process_documents(
    client: httpx.Client,
    url: str,
    index: str,
    num_chunks: int,
    segment_by_shard: list[tuple[int, int]],
) -> list[dict]:
    results = []
    collected_chunks = 0
    pbar = tqdm(range(num_chunks), desc="Retrieving documents from Infinigram")

    seen_documents = set()
    seen_chunks = set()

    for shard_index, (shard_start, shard_end) in enumerate(segment_by_shard):
        for rank in range(shard_start, shard_end):
            doc_data = get_document_by_rank(client, url, index, shard_index, rank)
            if not doc_data:
                continue
            doc_id = json.loads(doc_data.get("metadata")).get("metadata", {}).get("id")
            if doc_id in seen_documents:
                continue
            seen_documents.add(doc_id)

            # logger.debug(f"Processing document:\n{doc_data['metadata']['id']}")
            processed_doc = process_document(doc_data, rank)
            if processed_doc:
                for chunk in processed_doc["json_chunks"]:
                    chunk_str = json.dumps(chunk, ensure_ascii=False)
                    if chunk_str in seen_chunks:
                        continue  # if we saw this chunk before, skip the whole document
                    seen_chunks.add(chunk_str)
                results.append(processed_doc)
                collected_chunks += len(processed_doc["json_chunks"])
                pbar.set_postfix({"json_chunks": collected_chunks})
                pbar.update(1)
            if collected_chunks >= num_chunks:
                break
        if collected_chunks >= num_chunks:
            break
    return results


def get_json_chunks_infinigram(
    infinigram_url: str,
    infinigram_index: str,
    num_chunks: int,
) -> list[dict]:
    query = "```json"
    client = httpx.Client()
    find_results = find_documents(client, infinigram_url, infinigram_index, query)

    logger.debug(f"{find_results=}")

    doc_count = find_results.get("cnt", 0)
    if doc_count == 0:
        logger.info("No documents found matching the query.")
        return []

    logger.info(f"Found {doc_count} potential documents.")

    segment_by_shard = find_results["segment_by_shard"]
    return fetch_and_process_documents(
        client,
        url=infinigram_url,
        index=infinigram_index,
        num_chunks=num_chunks,
        segment_by_shard=segment_by_shard,
    )


def save_chunks_to_db(db_session, results: list[dict], meta: dict):
    new_documents = []
    new_json_chunks = []
    for result in results:
        document = result["document"]
        json_chunks = result["json_chunks"]
        document_meta = result["document_meta"]
        document_meta.update(meta)

        document_entity = SourceDocument(
            source="infinigram",
            content=document,
            content_type=ContentType.TEXT.value,
            meta=document_meta,
        )

        for json_chunk in json_chunks:
            json_chunk_entity = Chunk(
                content=json.dumps(json_chunk, indent=2, ensure_ascii=False),
                content_type=ContentType.JSON.value,
                is_synthetic=False,
                parent_document=document_entity,
                meta=document_meta,
            )
            new_json_chunks.append(json_chunk_entity)
        new_documents.append(document_entity)
    logger.info(
        f"Prepared {len(new_json_chunks)} json chunks and {len(new_documents)} documents."
    )

    deduplicated_chunks = deduplicate_chunks(new_json_chunks)
    db_session.add_all(deduplicated_chunks)
    db_session.commit()


def preview_chunks(results: list[dict]):
    for result in results:
        print(f"{result['document_meta']=}")
        print(f"{result['document']=}")
        print()
        for json_chunk in result["json_chunks"]:
            print(f"{json.dumps(json_chunk, indent=2, ensure_ascii=False)}")
            print()
        print()
        print()
        print()


@click.command()
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(),
    required=True,
    help="Sqlite3 file to save the database to",
)
@click.option(
    "--num-chunks",
    default=10,
    type=int,
    help="Number of chunks to retrieve",
)
@click.option(
    "--preview",
    is_flag=True,
    help="Preview the generated chunks, don't save to database",
)
@click.option(
    "--infinigram-url",
    default="https://api.infini-gram.io/",
    type=str,
    help="URL of the Infogram API",
)
@click.option(
    "--infinigram-index",
    default="v4_olmo-2-0325-32b-instruct_llama",
    type=str,
)
def run(
    db_file: str,
    num_chunks: int,
    preview: bool,
    infinigram_url: str,
    infinigram_index: str,
):
    logger.info(f"Generating input chunks from {db_file}")

    db_session = get_db_session(f"sqlite:///{db_file}")

    results = get_json_chunks_infinigram(
        infinigram_url=infinigram_url,
        infinigram_index=infinigram_index,
        num_chunks=num_chunks,
    )
    if not results:
        return

    if preview:
        preview_chunks(results)
        return

    try:
        meta = {
            "source": "infinigram",
            "infinigram_url": infinigram_url,
            "infinigram_index": infinigram_index,
        }
        save_chunks_to_db(db_session, results, meta)
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        raise e
    finally:
        db_session.close()


if __name__ == "__main__":
    load_dotenv(override=True)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    logger.info("Starting script")
    run()
