import json
import fastjsonschema
from sqlalchemy.orm import Session


from any2json.database.models import Chunk, SourceDocument
from any2json.enums import ContentType

from any2json.schema_utils import to_supported_json_schema

from any2json.data_engine.generators.base import SampleGenerator

from any2json.data_engine.generators.vary_schema import VaryJSONSchemaGenerator
from any2json.database.models import Chunk, JsonSchema, SchemaConversion
from any2json.enums import ContentType
from any2json.utils import logger
from tqdm.auto import tqdm


def get_json_chunks_with_schema(db_session: Session) -> list[Chunk]:
    return (
        db_session.query(Chunk)
        .filter(Chunk.schema_id.isnot(None))
        .filter(Chunk.content_type == ContentType.JSON.value)
        .filter(Chunk.is_synthetic == False)
        .all()
    )


def get_json_chunks_with_schema(db_session: Session) -> list[Chunk]:
    return (
        db_session.query(Chunk)
        .filter(Chunk.schema_id.isnot(None))
        .filter(Chunk.content_type == ContentType.JSON.value)
        .all()
    )


def get_json_documents(db_session: Session) -> list[SourceDocument]:
    return (
        db_session.query(SourceDocument)
        .filter(SourceDocument.content_type == ContentType.JSON.value)
        .all()
    )


def get_chunks_from_record(
    record: dict | list,
    max_depth: int = 3,
) -> list[dict | list]:
    """
    Get all the chunks from an arbitrary nested dictionary that might contain lists and dictionaries.

    Will recursively traverse the dictionary, pick out values that are not primitive types and return a list of InputJSONChunk objects.

    Args:
        record: A dictionary that might contain lists and dictionaries.

    Returns:
        A list of json-like objects.
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

        chunks.append(record)

    if max_depth == 0:
        return chunks

    if isinstance(record, dict):
        for k, v in record.items():
            if isinstance(v, (list, dict)):
                chunks.extend(get_chunks_from_record(v, max_depth - 1))
    elif isinstance(record, list):
        for v in record:
            chunks.extend(get_chunks_from_record(v, max_depth - 1))

    return chunks


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen_hashes = set()

    deduped_chunks = []
    for chunk in chunks:
        hash_value = hash(chunk.content.lower().strip())
        if hash_value not in seen_hashes:
            seen_hashes.add(hash_value)
            deduped_chunks.append(chunk)

    return deduped_chunks


def extract_json_chunks(
    documents: list[SourceDocument],
    max_depth: int = 3,
) -> list[Chunk]:
    chunks = []
    for document in documents:
        json_content = json.loads(document.content)

        chunk_jsons = get_chunks_from_record(json_content, max_depth=max_depth)

        for chunk_json in chunk_jsons:
            chunks.append(
                Chunk(
                    parent_document_id=document.id,
                    content=json.dumps(chunk_json),
                    content_type=ContentType.JSON.value,
                    is_synthetic=False,
                )
            )

    return chunks


import random
import click
import logging
import json
import os
from tqdm.auto import tqdm
import instructor
from any2json.data_engine.agents import (
    JSONSchemaValidationAgent,
    SchemaAgentInputSchema,
)
from any2json.database.client import get_db_session
from any2json.database.models import Chunk, JsonSchema
from any2json.enums import ContentType
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_json_chunks_with_no_schema(db_session: Session) -> list[Chunk]:
    return (
        db_session.query(Chunk)
        .filter(Chunk.schema_id.is_(None))
        .filter(Chunk.content_type == ContentType.JSON.value)
        .all()
    )


def generate_schema_for_json(
    json_content: dict,
    schema_agent: JSONSchemaValidationAgent,
) -> dict:
    original_data = json_content

    input_string = json.dumps(original_data, indent=1)

    try:
        schema = schema_agent.generate_and_validate_schema(
            SchemaAgentInputSchema(input_string=input_string)
        )
        simplified_schema = to_supported_json_schema(schema)
        validate = fastjsonschema.compile(simplified_schema)
        validate(json_content)

        if json.dumps(simplified_schema) != json.dumps(schema):
            logger.warning(
                f"Simplified schema differs from original schema. Original: {schema}, Simplified: {simplified_schema}"
            )
        schema = simplified_schema
    except Exception as e:
        logger.error(e, exc_info=True)
        return None

    return schema


def generate_schemas_for_chunks(
    db_session: Session,
    chunks: list[Chunk],
    schema_agent: JSONSchemaValidationAgent,
) -> tuple[list[JsonSchema], list[Chunk]]:
    schemas = []
    updated_chunks = []
    for i, chunk in tqdm(
        enumerate(chunks),
        desc="Generating schemas for chunks",
        total=len(chunks),
    ):
        schema = generate_schema_for_json(
            json.loads(chunk.content),
            schema_agent,
        )
        if schema:
            schema_entity = JsonSchema(
                content=schema,
                chunks=[chunk],
            )
            schemas.append(schema_entity)
            chunk.schema = schema_entity
            updated_chunks.append(chunk)

            try:
                db_session.add(schema_entity)
                db_session.add(chunk)
                db_session.commit()
            except Exception as e:
                logger.error(
                    f"Error: {e}\nFailed to commit chunk {chunk.id} schema: {schema_entity.content}",
                    exc_info=True,
                )
                raise e

    return schemas, updated_chunks


def generate_synthetic_chunks(
    db_session: Session,
    num_chunks: int,
    generator_class: type[SampleGenerator],
    **kwargs,
):
    input_chunks = []
    schemas = []
    output_chunks = []
    schema_conversions = []
    for i in tqdm(range(num_chunks)):
        generator = generator_class(**kwargs)
        generator.setup()
        generator_state = generator.get_state()
        input_format = ContentType(generator.format_name.upper()).value
        input_str, schema, output_json = generator.generate_triplet()

        meta = {
            "generator": generator.__class__.__name__,
            "generator_state": generator_state,
        }

        schema_entity = JsonSchema(
            content=schema,
            is_synthetic=True,
            meta=meta,
        )

        output_chunk_entity = Chunk(
            content=output_json,
            content_type=ContentType.JSON.value,
            schema=schema_entity,
            meta=meta,
            is_synthetic=True,
        )

        input_chunk_entity = Chunk(
            content=input_str,
            content_type=input_format,
            meta=meta,
            is_synthetic=True,
            parent_chunk_id=output_chunk_entity.id,
            matches_parent_chunk=True,
        )

        schema_conversion_entity = SchemaConversion(
            input_chunk=input_chunk_entity,
            schema=schema_entity,
            output_chunk=output_chunk_entity,
            meta={
                "generator": generator.__class__.__name__,
            },
        )

        input_chunks.append(input_chunk_entity)
        schemas.append(schema_entity)
        output_chunks.append(output_chunk_entity)
        schema_conversions.append(schema_conversion_entity)

        db_session.add(schema_entity)
        db_session.add(input_chunk_entity)
        db_session.add(output_chunk_entity)
        db_session.add(schema_conversion_entity)

    logger.info(f"Generated {len(input_chunks)} input chunks")
    return input_chunks, schemas, output_chunks, schema_conversions


def generate_synthetic_schemas(
    chunks: list[Chunk],
    num_generations: int | None = None,
    num_variations_per_schema: int = 1,
) -> list[Chunk]:
    new_chunks = []
    new_schemas = []
    new_schema_conversions = []

    total = num_generations or len(chunks) * num_variations_per_schema

    pbar = tqdm(total=total)

    for chunk in chunks:
        schema = chunk.schema
        json_content = (
            json.loads(chunk.content)
            if isinstance(chunk.content, str)
            else chunk.content
        )
        schema_content = (
            json.loads(schema.content)
            if isinstance(schema.content, str)
            else schema.content
        )

        if schema_content.get("type") != "object":
            continue

        generator = VaryJSONSchemaGenerator()
        generator.setup()
        generator_state = generator.get_state()

        try:
            samples = generator.generate_samples(
                json_content,
                schema_content,
                num_samples=num_variations_per_schema,
            )
        except Exception as e:
            logger.debug(f"Error generating samples for chunk {chunk.id}: {e}")
            logger.debug(e, exc_info=True)
            continue

        for source_schema, source_data, new_schema, new_data, changes in samples:
            meta = {
                "generator": generator.__class__.__name__,
                "generator_state": generator_state,
                "changes": changes,
            }

            new_schema_entity = JsonSchema(
                content=new_schema,
                is_synthetic=True,
                meta=meta,
                parent_schema_id=schema.id,
            )
            new_schemas.append(new_schema_entity)

            new_chunk_entity = Chunk(
                content=json.dumps(new_data),
                content_type=ContentType.JSON.value,
                schema=new_schema_entity,
                parent_chunk_id=chunk.id,
                matches_parent_chunk=False,
                is_synthetic=True,
                meta=meta,
            )
            new_chunks.append(new_chunk_entity)

            schema_conversion_entity = SchemaConversion(
                input_chunk=chunk,
                schema=new_schema_entity,
                output_chunk=new_chunk_entity,
                meta=meta,
            )
            new_schema_conversions.append(schema_conversion_entity)

            pbar.update(1)

        if num_generations and len(new_schemas) >= num_generations:
            break
    pbar.close()

    return new_schemas, new_chunks, new_schema_conversions
