import logging
import random
import click
import os
import json

from sqlalchemy.orm import Session

from any2json.data_engine.generators.vary_schema import VaryJSONSchemaGenerator
from any2json.data_engine.utils import (
    deduplicate_chunks,
    get_chunks_from_record,
)
from any2json.database.client import get_db_session
from any2json.database.models import Chunk, JsonSchema, SourceDocument
from any2json.enums import ContentType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_json_chunks_with_schema(db_session: Session) -> list[Chunk]:
    return (
        db_session.query(Chunk)
        .filter(Chunk.schema_id.isnot(None))
        .filter(Chunk.content_type == ContentType.JSON.value)
        .filter(Chunk.is_synthetic == False)
        .all()
    )


def generate_synthetic_schemas(
    chunks: list[Chunk],
    num_generations: int | None = None,
    num_variations_per_schema: int = 1,
) -> list[Chunk]:
    new_chunks = []
    new_schemas = []

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
                is_synthetic=True,
                meta=meta,
            )
            new_chunks.append(new_chunk_entity)

            if num_generations and len(new_schemas) >= num_generations:
                break
        if num_generations and len(new_schemas) >= num_generations:
            break

    return new_schemas, new_chunks


@click.command()
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(),
    required=True,
    help="Sqlite3 file to save the database to",
)
@click.option(
    "--num-variations-per-schema",
    default=1,
    type=int,
    required=True,
)
@click.option(
    "--num-generations",
    default=None,
    type=int,
    required=False,
)
@click.option(
    "--preview",
    is_flag=True,
    help="Preview the generated chunks, don't save to database",
)
def run(
    db_file: str,
    num_variations_per_schema: int,
    num_generations: int | None,
    preview: bool,
):
    logger.info(f"Generating input chunks from {db_file}")

    db_session = get_db_session(f"sqlite:///{db_file}")

    try:
        chunks = get_json_chunks_with_schema(db_session)
        random.shuffle(chunks)

        new_schemas, new_chunks = generate_synthetic_schemas(
            chunks,
            num_generations=num_generations,
            num_variations_per_schema=num_variations_per_schema,
        )
        logger.info(f"Generated {len(new_schemas)} synthetic schemas")
        logger.info(f"Generated {len(new_chunks)} synthetic chunks")

        deduplicated_chunks = deduplicate_chunks(new_chunks)
        deduplicated_schemas = [c.schema for c in deduplicated_chunks]

        logger.info(f"Deduplicated {len(deduplicated_chunks)} synthetic chunks")
        logger.info(f"Deduplicated {len(deduplicated_schemas)} synthetic schemas")

        db_session.add_all(deduplicated_schemas)
        db_session.add_all(deduplicated_chunks)

        if preview:
            for chunk in deduplicated_chunks:
                print(f"{chunk.content=}")
                print(f"{chunk.meta=}")
                print()
                print(f"{chunk.schema.content=}")
                print()
                print(f"{chunk.schema.meta=}")
                print()
                print()
            logger.info("Previewed synthetic chunks, not saving to database")
        else:
            db_session.commit()
            logger.info("Committed to database")
    except Exception as e:
        db_session.rollback()
        raise e


if __name__ == "__main__":
    run()
