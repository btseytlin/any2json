from collections.abc import Callable
from functools import partial
import fastjsonschema
from sqlalchemy.orm import Session

import logging
import click
import os
import json
from datasets import Dataset
from dataclasses import asdict
import toml
from tqdm import tqdm
import yaml

from any2json.containers import InputJSONChunk
from any2json.data_engine.helpers import deduplicate_chunks, expand_refs_in_schema
from any2json.database.client import create_tables, get_db_session
from any2json.database.models import Chunk, JsonSchema, SchemaConversion, SourceDocument
from any2json.encoder import DateTimeEncoder
from any2json.enums import ContentType
from any2json.utils import logger
from any2json.schema_utils import to_supported_json_schema


def validate_schema_and_response(
    schema: dict | str,
    response: dict | str,
) -> bool:
    if isinstance(schema, str):
        schema = json.loads(schema)

    if isinstance(response, str):
        response = json.loads(response)

    validator = fastjsonschema.compile(schema, use_formats=False)
    validator(response)


def validate_schema_quality(schema: dict) -> bool:
    if not schema:
        return False

    if not schema.get("type"):
        return False

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if schema_type[0] not in ["object", "array"]:
            return False
    elif schema_type not in ["object", "array"]:
        return False

    if schema_type == "array" and not schema.get("items", {}).get("type"):
        return False

    if not schema.get("properties") and not schema.get("items"):
        return False

    return True


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

            response = json.loads(record["response"])
            schema = json.loads(record["schema"])
            schema = expand_refs_in_schema(schema, schema)
            schema = to_supported_json_schema(schema)

            validate_schema_and_response(schema, response)

            source_document = SourceDocument(
                source=dirname_to_dataset_id(dataset_dir),
                content="",
                content_type=ContentType.JSON.value,
                meta={
                    "source_dataset_index": i,
                },
            )
            db_session.add(source_document)

            chunk = Chunk(
                content=json.dumps(response),
                content_type=ContentType.JSON.value,
                is_synthetic=True,  # This dataset has LLM-generated respones and schemas, so they are not "real"
                parent_document_id=source_document.id,
            )
            db_session.add(chunk)

            assert validate_schema_quality(schema), f"Low quality schema: {schema}"
            json_schema = JsonSchema(
                content=schema,
                is_synthetic=True,
                chunks=[chunk],
                meta={
                    "source_dataset": dirname_to_dataset_id(dataset_dir),
                    "source_dataset_index": i,
                    "description": record.get("description"),
                },
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

    logger.info(f"Processed {i} records, {errors} errors, {i - errors} success")


def processor_interstellarninja_json_mode_reasoning(
    db_session: Session,
    dataset_dir: str,
    num_samples_per_dataset: int | None = None,
    max_length: int = 10000,
) -> None:
    dataset = Dataset.load_from_disk(dataset_dir)

    errors = 0
    skips = 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    seen_chunks = set()
    seen_schemas = set()
    chunks = []
    json_schemas = []
    source_documents = []
    for i, record in pbar:
        try:
            response_text = record["conversations"][-1]["value"]
            response_text = response_text.split("</think>")[-1].strip()

            if len(response_text) > max_length:
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

            if hash(response_text) in seen_chunks:
                logger.debug(f"Skipping record {i} because response is already seen")
                skips += 1
                continue

            if hash(record["schema"]) in seen_schemas:
                logger.debug(f"Skipping record {i} because schema is already seen")
                skips += 1
                continue

            response = json.loads(response_text)

            schema_original = json.loads(record["schema"])
            schema_supported = to_supported_json_schema(
                schema_original, expand_refs=True
            )

            if not validate_schema_quality(schema_supported):
                logger.debug(f"Skipping record {i} because schema is low quality")
                skips += 1
                continue

            validator_original = fastjsonschema.compile(
                schema_original,
                use_formats=False,
            )
            validator_supported = fastjsonschema.compile(
                schema_supported,
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

            json_schema = JsonSchema(
                content=schema_supported,
                is_synthetic=True,
                chunks=[chunk],
                meta={
                    "source_dataset": dirname_to_dataset_id(dataset_dir),
                    "source_dataset_index": i,
                    "description": record.get("description"),
                },
            )

            seen_chunks.add(hash(response_text))
            seen_schemas.add(hash(record["schema"]))

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

    logger.info(
        f"Processed {i} records, {errors} errors, {skips} skips, {i - errors - skips} success"
    )

    db_session.add_all(chunks)
    db_session.add_all(json_schemas)
    db_session.add_all(source_documents)


def processor_dataunitylab_json_schema(
    db_session: Session,
    dataset_dir: str,
    num_samples_per_dataset: int | None = None,
    max_length: int = 10000,
    schema_key: str = "content",
    description_key: str = "description",
) -> None:
    dataset = Dataset.load_from_disk(dataset_dir)

    errors = 0
    skips = 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    seen_schemas = set()
    json_schemas = []
    source_documents = []
    for i, record in pbar:
        try:
            schema_text = record[schema_key]

            description = record.get(description_key)

            if len(schema_text) > max_length:
                logger.debug(
                    f"Skipping record {i} because schema length is {len(schema_text)} > {max_length}"
                )
                skips += 1
                continue

            if hash(schema_text) in seen_schemas:
                logger.debug(f"Skipping record {i} because schema is already seen")
                skips += 1
                continue

            schema_original = json.loads(schema_text)

            schema_supported = to_supported_json_schema(schema_original)

            if not validate_schema_quality(schema_supported):
                logger.debug(f"Skipping record {i} because schema is low quality")
                skips += 1
                continue

            fastjsonschema.compile(
                schema_original,
                use_formats=False,
            )
            fastjsonschema.compile(
                schema_supported,
                use_formats=False,
            )

            metadata = {
                "source_dataset_index": i,
            }

            source_document = SourceDocument(
                source=dirname_to_dataset_id(dataset_dir),
                content="",
                content_type=ContentType.JSON.value,
                meta=metadata,
            )

            json_schema = JsonSchema(
                content=schema_supported,
                is_synthetic=True,
                meta={
                    "source_dataset": dirname_to_dataset_id(dataset_dir),
                    "source_dataset_index": i,
                    "description": description,
                },
            )

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

    logger.info(
        f"Processed {i} records, {errors} errors, {skips} skips, {i - errors - skips} success"
    )

    db_session.add_all(json_schemas)
    db_session.add_all(source_documents)


def processor_schemastore_schemastore(
    db_session: Session,
    dataset_dir: str,
    num_samples_per_dataset: int | None = None,
) -> None:
    schemas_dir = os.path.join(dataset_dir, "src", "schemas", "json")
    test_dir = os.path.join(dataset_dir, "src", "test")

    schemas_created = 0
    chunks_created = 0
    schema_conversions_created = 0

    total = num_samples_per_dataset or len(os.listdir(schemas_dir))

    pbar = tqdm(
        os.listdir(schemas_dir),
        total=total,
        desc="Processing schemas",
    )

    for schema_file in pbar:
        try:
            schema_name = schema_file.split(".json")[0]
            schema_path = os.path.join(schemas_dir, schema_file)

            with open(schema_path, "r") as f:
                schema = json.load(f)

            validator_original = fastjsonschema.compile(
                schema,
                use_formats=False,
            )

            schema = to_supported_json_schema(schema, expand_refs=True)

            validator_processed = fastjsonschema.compile(
                schema,
                use_formats=False,
            )

            if not validate_schema_quality(schema):
                logger.debug(f"Skipping schema {schema_name} because it is low quality")
                continue

            meta = {
                "source_dataset": dirname_to_dataset_id(dataset_dir),
                "schema_name": schema_name,
            }

            json_schema = JsonSchema(
                content=schema,
                is_synthetic=False,
                meta=meta,
            )
            db_session.add(json_schema)
            schemas_created += 1

            tests_dir = os.path.join(test_dir, schema_name)
            if not os.path.exists(tests_dir):
                logger.debug(
                    f"Skipping schema {schema_name} tests because dir {tests_dir} does not exist"
                )
                continue

            for test_file in os.listdir(tests_dir):
                test_path = os.path.join(tests_dir, test_file)

                chunk_meta = meta.copy()
                chunk_meta["file_path"] = test_path

                if test_file.endswith(".json"):
                    with open(test_path, "r") as f:
                        test_data = json.load(f)

                    validator_original(test_data)
                    validator_processed(test_data)

                    chunk = Chunk(
                        content=json.dumps(test_data),
                        content_type=ContentType.JSON.value,
                        is_synthetic=False,
                        meta=chunk_meta,
                    )
                    json_schema.chunks.append(chunk)
                    chunks_created += 1
                elif test_file.endswith(".yaml") or test_file.endswith(".yml"):
                    with open(test_path, "r") as f:
                        test_data = f.read()

                    chunk = Chunk(
                        content=test_data,
                        content_type=ContentType.YAML.value,
                        is_synthetic=False,
                        meta=chunk_meta,
                    )
                    json_schema.chunks.append(chunk)
                    chunks_created += 1
                    json_chunk = yaml.safe_load(test_data)

                    validator_original(json_chunk)
                    validator_processed(json_chunk)

                    json_chunk = json.dumps(json_chunk, cls=DateTimeEncoder)

                    json_chunk_chunk = Chunk(
                        content=json_chunk,
                        content_type=ContentType.JSON.value,
                        is_synthetic=False,
                        meta=chunk_meta,
                    )
                    json_schema.chunks.append(json_chunk_chunk)
                    chunks_created += 1
                    schema_conversion = SchemaConversion(
                        input_chunk=chunk,
                        schema=json_schema,
                        output_chunk=json_chunk_chunk,
                    )
                    db_session.add(schema_conversion)
                    schema_conversions_created += 1
                elif test_file.endswith(".toml"):
                    with open(test_path, "r") as f:
                        test_data = f.read()

                    chunk = Chunk(
                        content=test_data,
                        content_type=ContentType.TOML.value,
                        is_synthetic=False,
                        meta=chunk_meta,
                    )
                    json_schema.chunks.append(chunk)
                    chunks_created += 1
                    json_chunk = toml.loads(test_data)

                    validator_original(json_chunk)
                    validator_processed(json_chunk)

                    json_chunk = json.dumps(json_chunk, cls=DateTimeEncoder)

                    json_chunk_chunk = Chunk(
                        content=json_chunk,
                        content_type=ContentType.JSON.value,
                        is_synthetic=False,
                        meta=chunk_meta,
                    )
                    json_schema.chunks.append(json_chunk_chunk)
                    chunks_created += 1
                    schema_conversion = SchemaConversion(
                        input_chunk=chunk,
                        schema=json_schema,
                        output_chunk=json_chunk_chunk,
                    )
                    db_session.add(schema_conversion)
                    schema_conversions_created += 1
                else:
                    raise RuntimeError(
                        f"Dont know how to handle file type: {test_file}"
                    )
        except Exception as e:
            logger.error(f"Error processing schema {schema_name}: {e}", exc_info=True)
            # raise e

        if num_samples_per_dataset and schemas_created >= num_samples_per_dataset:
            break

        pbar.set_postfix(
            {
                "schemas_created": schemas_created,
                "chunks_created": chunks_created,
                "schema_conversions_created": schema_conversions_created,
            }
        )

    logger.info(
        f"Processed {len(os.listdir(schemas_dir))} schemas, {schemas_created} schemas created, {chunks_created} chunks created, {schema_conversions_created} schema conversions created"
    )


def get_dataset_processor(input_dir: str) -> Callable:
    dataset_processors = {
        "wikimedia/structured-wikipedia": processor_wikimedia_structured_wikipedia,
        "christianazinn/json-training": processor_christian_azinn_json_training,
        "interstellarninja/json-mode-reasoning": processor_interstellarninja_json_mode_reasoning,
        "interstellarninja/json-mode-verifiable": processor_interstellarninja_json_mode_reasoning,
        "interstellarninja/json-mode-agentic-reasoning": processor_interstellarninja_json_mode_reasoning,
        "interstellarninja/json-schema-store-reasoning": processor_interstellarninja_json_mode_reasoning,  # high error rate, todo debug
        "dataunitylab/json-schema": partial(
            processor_dataunitylab_json_schema, schema_key="content"
        ),
        "dataunitylab/json-schema-descriptions": partial(
            processor_dataunitylab_json_schema, schema_key="object"
        ),
        "schemastore/schemastore": processor_schemastore_schemastore,
    }

    dataset_id = dirname_to_dataset_id(input_dir)
    return dataset_processors[dataset_id]
