import asyncio
import json
import fastjsonschema
from sqlalchemy import func, select
from sqlalchemy.orm import Session


from any2json.database.models import Chunk, SourceDocument
from any2json.enums import ContentType

from any2json.index import IndexedJsonSet
from any2json.schema_utils import to_supported_json_schema

from any2json.data_engine.generators.base import SampleGenerator

from any2json.data_engine.generators.vary_schema import VaryJSONSchemaGenerator
from any2json.database.models import Chunk, JsonSchema, SchemaConversion
from any2json.enums import ContentType
from any2json.utils import logger
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio


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


def get_chunk_existing_conversion_formats(
    db_session: Session,
) -> dict[int, list[ContentType]]:
    query = (
        select(
            SchemaConversion.output_chunk_id,
            func.json_group_array(Chunk.content_type).label("content_types"),
        )
        .join(Chunk, SchemaConversion.input_chunk)
        .group_by(SchemaConversion.output_chunk_id)
    )
    result = db_session.execute(query).all()
    result = {
        output_chunk_id: [
            ContentType(content_type) for content_type in json.loads(content_types)
        ]
        for output_chunk_id, content_types in result
    }
    return result


def get_json_documents(db_session: Session) -> list[SourceDocument]:
    return (
        db_session.query(SourceDocument)
        .filter(SourceDocument.content_type == ContentType.JSON.value)
        .filter(SourceDocument.content != "")
        .filter(SourceDocument.content is not None)
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

    if max_depth <= 0:
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
    max_depth: int = 7,
    frac_per_document: float = 0.2,
    max_chunks: int | None = None,
) -> list[Chunk]:
    chunks = []
    for document in documents:
        json_content = json.loads(document.content)

        chunk_jsons = get_chunks_from_record(json_content, max_depth=max_depth)

        chunk_jsons_filtered = []
        for chunk_json in chunk_jsons:
            try:
                chunk_str = json.dumps(chunk_json)
                if len(chunk_str) >= 100:
                    chunk_jsons_filtered.append(chunk_json)
            except Exception as e:
                continue

        chunk_jsons = random.sample(
            chunk_jsons_filtered,
            max(int(len(chunk_jsons_filtered) * frac_per_document), 1),
        )

        for chunk_json in chunk_jsons:
            chunks.append(
                Chunk(
                    parent_document_id=document.id,
                    content=json.dumps(chunk_json),
                    content_type=ContentType.JSON.value,
                    is_synthetic=False,
                )
            )

        if max_chunks and len(chunks) >= max_chunks:
            break

    return chunks


import random
import click
import logging
import json
import os
from tqdm.auto import tqdm
import instructor
from any2json.agents.schema_generator import JSONSchemaGeneratorAgent
from any2json.agents.chunk_generator import JSONChunkGeneratorAgent
from any2json.database.models import Chunk, JsonSchema
from any2json.enums import ContentType
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_json_chunks_with_no_schema(
    db_session: Session,
    limit: int | None = None,
    offset: int | None = None,
) -> list[Chunk]:
    return (
        db_session.query(Chunk)
        .filter(Chunk.schema_id.is_(None))
        .filter(Chunk.content_type == ContentType.JSON.value)
        .limit(limit)
        .offset(offset)
        .all()
    )


def get_schemas_with_no_chunks(db_session: Session) -> list[JsonSchema]:
    query = select(JsonSchema).where(JsonSchema.chunks.is_(None))
    return db_session.execute(query).scalars().all()


def check_if_schema_for_json_exists_in_db(
    json_content: dict | list | str | int | float | bool | None,
    index: IndexedJsonSet,
) -> bool:
    result = index.query(json_content, k=1)
    schema, schema_id, score = result[0]
    fastjsonschema.validate(json_content, schema, detailed_exceptions=False)
    return schema_id


def map_chunks_to_existing_schemas(
    db_session: Session,
    chunks: list[Chunk],
) -> list[int | None]:
    existing_schemas = db_session.query(JsonSchema).all()
    index = IndexedJsonSet([(e.content, e.id) for e in existing_schemas])

    results = []

    for chunk in tqdm(chunks):
        json_content = json.loads(chunk.content)
        try:
            schema_id = check_if_schema_for_json_exists_in_db(
                json_content,
                index,
            )
            results.append(schema_id)
        except Exception as e:
            results.append(None)

    return results


async def generate_schema_for_json(
    json_content: dict,
    schema_agent: JSONSchemaGeneratorAgent,
) -> tuple[dict, str]:
    try:
        schema, model_info = await schema_agent.generate_and_validate_schema(
            input_json=json_content,
        )
        simplified_schema = to_supported_json_schema(schema)
        validate = fastjsonschema.compile(simplified_schema)
        validate(json_content)

        if json.dumps(simplified_schema) != json.dumps(schema):
            logger.warning(
                f"Simplified schema differs from original schema. Original: {schema}, Simplified: {simplified_schema}"
            )
        schema = simplified_schema
        return schema, model_info
    except Exception as e:
        logger.debug(f"Error generating schema for {json_content}: {e}", exc_info=True)
        return None, None


async def generate_schemas_for_chunks(
    db_session: Session,
    chunks: list[Chunk],
    schema_agent: JSONSchemaGeneratorAgent,
) -> tuple[list[JsonSchema], list[Chunk]]:
    schemas_generated = 0
    updated_chunks = 0

    async def generate_schema_for_json_wrapper(
        json_content: dict,
        agent: JSONSchemaGeneratorAgent,
        chunk_id: int,
    ):
        return chunk_id, await generate_schema_for_json(
            json_content,
            agent,
        )

    chunks_lookup = {chunk.id: chunk for chunk in chunks}
    tasks = []
    for chunk in chunks:
        tasks.append(
            generate_schema_for_json_wrapper(
                json.loads(chunk.content),
                agent=schema_agent,
                chunk_id=chunk.id,
            )
        )

    for result in tqdm_asyncio.as_completed(tasks):
        chunk_id, (schema, model_name) = await result
        chunk = chunks_lookup[chunk_id]

        if not schema:
            continue

        schema_entity = JsonSchema(
            content=schema,
            chunks=[chunk],
            meta={
                "generator_state": schema_agent.get_state(),
                "model_name": model_name,
            },
        )
        db_session.add(schema_entity)

        chunk.schema = schema_entity

        db_session.add(chunk)
        db_session.commit()

        schemas_generated += 1
        updated_chunks += 1

    return schemas_generated, updated_chunks


async def generate_chunks_for_schemas(
    db_session: Session,
    schemas: list[JsonSchema],
    agent: JSONChunkGeneratorAgent,
) -> int:
    chunks_generated = 0
    updated_schemas = 0
    errors = 0

    async def generate_chunk_for_schema_wrapper(
        input_schema: dict,
        agent: JSONChunkGeneratorAgent,
        schema_id: int,
    ):
        return schema_id, await agent.generate_and_validate_json(
            input_schema=input_schema,
        )

    schemas_lookup = {schema.id: schema for schema in schemas}
    tasks = []
    for schema in schemas:
        tasks.append(
            generate_chunk_for_schema_wrapper(
                input_schema=schema.content,
                agent=agent,
                schema_id=schema.id,
            )
        )

    for result in tqdm_asyncio.as_completed(tasks):
        try:
            schema_id, (json_chunk, model_name) = await result
            schema = schemas_lookup[schema_id]
            if not json_chunk:
                continue
            chunk = Chunk(
                content=json.dumps(json_chunk),
                content_type=ContentType.JSON.value,
                schema=schema,
                is_synthetic=True,
                meta={
                    "generator_state": agent.get_state(),
                    "model_name": model_name,
                },
            )
            db_session.add(chunk)
            schema.chunks.append(chunk)
            db_session.add(schema)
            db_session.commit()

            chunks_generated += 1
            updated_schemas += 1
        except Exception as e:
            logger.error(
                f"Error generating chunk for schema {schema.id}: {e}", exc_info=True
            )
            continue

    return chunks_generated, updated_schemas, errors


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
        try:
            generator = generator_class(**kwargs)
            generator.setup()
            generator_state = generator.get_state()
            input_format = ContentType(generator.input_format.upper()).value
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
                meta=meta,
            )

            input_chunks.append(input_chunk_entity)
            schemas.append(schema_entity)
            output_chunks.append(output_chunk_entity)
            schema_conversions.append(schema_conversion_entity)

            db_session.add(schema_entity)
            db_session.add(input_chunk_entity)
            db_session.add(output_chunk_entity)
            db_session.add(schema_conversion_entity)
        except Exception as e:
            logger.error(e, exc_info=True)
            continue

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
