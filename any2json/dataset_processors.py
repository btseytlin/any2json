from collections.abc import Callable
import fastjsonschema
from sqlalchemy.orm import Session

import logging
import click
import os
import json
from datasets import Dataset
from dataclasses import asdict

from any2json.containers import InputJSONChunk
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
    for i, record in enumerate(dataset):
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
    for i, record in enumerate(dataset):
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
                content=json.dumps(record),
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
                logger.debug("Original record: " + json.dumps(record, indent=1))
                logger.debug("Schema: " + json.dumps(schema, indent=1))
                logger.debug(
                    "Response: " + json.dumps(json.loads(record["response"]), indent=1)
                )
            except Exception:
                pass
            errors += 1

    logger.info(f"Processed {i} records, {errors} errors")


def get_dataset_processor(input_dir: str) -> Callable:
    dataset_processors = {
        "wikimedia/structured-wikipedia": processor_wikimedia_structured_wikipedia,
        "christianazinn/json-training": processor_christian_azinn_json_training,
    }

    dataset_id = dirname_to_dataset_id(input_dir)
    return dataset_processors[dataset_id]
