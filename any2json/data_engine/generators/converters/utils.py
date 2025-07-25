import random
from any2json.data_engine.helpers import deduplicate_chunks
from any2json.database.models import Chunk, SchemaConversion
from any2json.data_engine.generators.converters.converters import (
    ToYamlConverter,
    ToTomlConverter,
    ToMarkdownTableConverter,
    ToPythonStringConverter,
)
from any2json.data_engine.helpers import get_json_chunks_with_schema
from any2json.utils import logger
import json
from sqlalchemy.orm import Session


def generate_format_converted_chunks(
    chunks: list[Chunk],
    num_generations: int | None = None,
) -> list[Chunk]:
    new_chunks = []
    schema_conversions = []

    for chunk in chunks:
        schema = chunk.schema
        json_content = (
            json.loads(chunk.content)
            if isinstance(chunk.content, str)
            else chunk.content
        )

        converters = [
            ToYamlConverter,
            ToTomlConverter,
            # ToMarkdownTableConverter, # Broken it seems
            ToPythonStringConverter,
        ]

        for converter in converters:
            try:
                converter = converter()
                converter.setup()
                converter_state = converter.get_state()

                new_chunk_string = converter.convert(json_content)

                assert new_chunk_string.strip(), "New chunk string is empty"

                meta = {
                    "converter": converter.__class__.__name__,
                    "converter_state": converter_state,
                }

                new_chunk_entity = Chunk(
                    content=new_chunk_string,
                    content_type=converter.format.value,
                    parent_chunk_id=chunk.id,
                    matches_parent_chunk=True,
                    schema=schema,
                    is_synthetic=True,
                    meta=meta,
                )

                schema_conversion_entity = SchemaConversion(
                    input_chunk=new_chunk_entity,
                    schema=schema,
                    output_chunk=chunk,
                    meta=meta,
                )

            except Exception as e:
                logger.warning(
                    f"Error converting chunk {chunk.id} to {converter.format}: {e}",
                    exc_info=True,
                )
                continue
            new_chunks.append(new_chunk_entity)
            schema_conversions.append(schema_conversion_entity)

        if num_generations and len(new_chunks) >= num_generations:
            break

    return new_chunks, schema_conversions


def generate_synthetic_format_conversions(
    db_session: Session,
    num_generations: int | None = None,
) -> list[Chunk]:
    chunks = get_json_chunks_with_schema(db_session)
    random.shuffle(chunks)

    new_chunks, schema_conversions = generate_format_converted_chunks(
        chunks,
        num_generations=num_generations,
    )
    logger.info(f"Generated {len(new_chunks)} synthetic chunks")

    deduplicated_chunks = deduplicate_chunks(new_chunks)

    deduplicated_chunks_ids = [chunk.id for chunk in deduplicated_chunks]

    schema_conversions = [
        schema_conversion
        for schema_conversion in schema_conversions
        if schema_conversion.input_chunk_id in deduplicated_chunks_ids
    ]

    logger.info(
        f"After deduplication: {len(deduplicated_chunks)} synthetic chunks, {len(schema_conversions)} schema conversions"
    )

    db_session.add_all(deduplicated_chunks)
    db_session.add_all(schema_conversions)
    return deduplicated_chunks, schema_conversions
