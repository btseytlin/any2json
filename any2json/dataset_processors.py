from collections.abc import Callable
import fastjsonschema
from sqlalchemy.orm import Session

import logging
import click
import os
import json
from datasets import Dataset
from dataclasses import asdict
from tqdm import tqdm

from any2json.containers import InputJSONChunk
from any2json.data_engine.helpers import deduplicate_chunks
from any2json.database.client import create_tables, get_db_session
from any2json.database.models import Chunk, JsonSchema, SourceDocument
from any2json.enums import ContentType
from any2json.utils import logger
from any2json.schema_utils import to_supported_json_schema


def dirname_to_dataset_id(dirname: str) -> str:
    return "/".join(dirname.split("/")[-2:]).lower()


def processor_wikimedia_structured_wikipedia(
    db_session: Session,
    dataset_dir: str,
    num_samples_per_dataset: int | None = None,
) -> None:
    dataset = Dataset.load_from_disk(dataset_dir)

    source_documents = []
    for i, record in tqdm(enumerate(dataset), total=len(dataset)):
        if isinstance(record, (list, dict)):
            source_documents.append(
                SourceDocument(
                    source=dirname_to_dataset_id(dataset_dir),
                    content=json.dumps(record),
                    content_type=ContentType.JSON.value,
                    meta={
                        "source_dataset_index": i,
                    },
                )
            )

        if num_samples_per_dataset and len(source_documents) >= num_samples_per_dataset:
            break

    logger.info(f"Adding {len(source_documents)} source documents")
    db_session.add_all(source_documents)


def processor_christian_azinn_json_training(
    db_session: Session,
    dataset_dir: str,
    num_samples_per_dataset: int | None = None,
) -> list[SourceDocument]:
    dataset = Dataset.load_from_disk(dataset_dir)

    errors = 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    for i, record in pbar:
        try:
            if "definitions" in record["schema"]:
                record["schema"] = record["schema"].replace("definitions", "$defs")

            schema = json.loads(record["schema"])
            schema = to_supported_json_schema(schema)
            response = json.loads(record["response"])

            validator = fastjsonschema.compile(schema)
            validator(response)

            metadata = {
                "source_dataset_index": i,
            }

            source_document = SourceDocument(
                source=dirname_to_dataset_id(dataset_dir),
                content="",
                content_type=ContentType.JSON.value,
                meta=metadata,
            )
            db_session.add(source_document)

            chunk = Chunk(
                content=json.dumps(response),
                content_type=ContentType.JSON.value,
                is_synthetic=True,  # This dataset has LLM-generated respones and schemas, so they are not "real"
                parent_document_id=source_document.id,
            )
            db_session.add(chunk)

            json_schema = JsonSchema(
                content=schema,
                is_synthetic=True,
                chunks=[chunk],
            )
            db_session.add(json_schema)

            if num_samples_per_dataset and i >= num_samples_per_dataset:
                break
        except Exception as e:
            logger.debug(f"Error processing record {i}: {e}")
            try:
                logger.debug("Schema: " + json.dumps(schema, indent=1))
                logger.debug(
                    "Response: " + json.dumps(json.loads(record["response"]), indent=1)
                )
            except Exception:
                pass
            errors += 1
        pbar.set_postfix({"errors": errors})

    logger.info(f"Processed {i} records, {errors} errors")


def processor_interstellarninja_json_mode_reasoning(
    db_session: Session,
    dataset_dir: str,
    num_samples_per_dataset: int | None = None,
) -> None:
    dataset = Dataset.load_from_disk(dataset_dir)

    max_length = 4000
    errors = 0
    skips = 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    seen_chunks = set()
    chunks = []
    json_schemas = []
    source_documents = []
    for i, record in pbar:
        try:
            response = record["conversations"][-1]["value"]
            response = response.split("</think>")[-1].strip()

            if len(response) > max_length:
                logger.debug(
                    f"Skipping record {i} because response length is {len(response)} > {max_length}"
                )
                skips += 1
                continue

            if len(record["schema"]) > max_length:
                logger.debug(
                    f"Skipping record {i} because schema length is {len(record['schema'])} > {max_length}"
                )
                skips += 1
                continue

            response = json.loads(response)

            schema_original = json.loads(record["schema"])
            schema_supported = to_supported_json_schema(schema_original)

            validator_original = fastjsonschema.compile(
                schema_original,
                detailed_exceptions=False,
                use_formats=False,
            )
            validator_supported = fastjsonschema.compile(
                schema_supported,
                detailed_exceptions=False,
                use_formats=False,
            )

            validator_original(response)
            validator_supported(response)

            metadata = {
                "source_dataset_index": i,
            }

            source_document = SourceDocument(
                source=dirname_to_dataset_id(dataset_dir),
                content="",
                content_type=ContentType.JSON.value,
                meta=metadata,
            )

            chunk = Chunk(
                content=json.dumps(response),
                content_type=ContentType.JSON.value,
                is_synthetic=True,  # This dataset has LLM-generated respones and schemas, so they are not "real"
                parent_document_id=source_document.id,
            )

            if response in seen_chunks:
                logger.debug(f"Skipping record {i} because response is already seen")
                skips += 1
                continue

            seen_chunks.add(response)

            json_schema = JsonSchema(
                content=schema_supported,
                is_synthetic=True,
                chunks=[chunk],
            )

            chunks.append(chunk)
            json_schemas.append(json_schema)
            source_documents.append(source_document)

            if num_samples_per_dataset and i >= num_samples_per_dataset:
                break
        except Exception as e:
            logger.debug(f"Error processing record {i}: {e}")
            # try:
            #     logger.debug(
            #         "Schema original: " + json.dumps(schema_original, indent=1)
            #     )
            #     logger.debug(
            #         "Schema supported: " + json.dumps(schema_supported, indent=1)
            #     )
            #     logger.debug("Response: " + json.dumps(response, indent=1))
            # except Exception:
            #     pass
            errors += 1
        pbar.set_postfix(
            {"error": errors, "skip": skips, "success": i - errors - skips}
        )

    logger.info(f"Processed {i} records, {errors} errors, {skips} skips")

    db_session.add_all(chunks)
    db_session.add_all(json_schemas)
    db_session.add_all(source_documents)


def get_dataset_processor(input_dir: str) -> Callable:
    dataset_processors = {
        "wikimedia/structured-wikipedia": processor_wikimedia_structured_wikipedia,
        "christianazinn/json-training": processor_christian_azinn_json_training,
        "interstellarninja/json-mode-reasoning": processor_interstellarninja_json_mode_reasoning,
    }

    dataset_id = dirname_to_dataset_id(input_dir)
    return dataset_processors[dataset_id]
