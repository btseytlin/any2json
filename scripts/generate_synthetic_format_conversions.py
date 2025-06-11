import logging
import random
import click
import json

from sqlalchemy.orm import Session

from any2json.data_engine.generators.converters.yaml import (
    ToHtmlTreeConverter,
    ToMarkdownTableConverter,
    ToPythonStringConverter,
    ToTomlConverter,
    ToYamlConverter,
)
from any2json.data_engine.utils import (
    deduplicate_chunks,
)
from any2json.database.client import get_db_session
from any2json.database.models import Chunk
from any2json.enums import ContentType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_json_chunks_with_schema(db_session: Session) -> list[Chunk]:
    return (
        db_session.query(Chunk)
        .filter(Chunk.schema_id.isnot(None))
        .filter(Chunk.content_type == ContentType.JSON.value)
        .all()
    )


def generate_format_converted_chunks(
    chunks: list[Chunk],
    num_generations: int | None = None,
) -> list[Chunk]:
    new_chunks = []

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
            ToMarkdownTableConverter,
            ToPythonStringConverter,
            ToHtmlTreeConverter,
        ]

        for converter in converters:
            try:
                converter = converter()
                converter.setup()
                converter_state = converter.get_state()

                new_chunk_string = converter.convert(json_content)

                meta = {
                    "converter": converter.__class__.__name__,
                    "converter_state": converter_state,
                }

                new_chunk_entity = Chunk(
                    content=new_chunk_string,
                    content_type=converter.format.value,
                    parent_chunk_id=chunk.id,
                    schema=schema,
                    is_synthetic=True,
                    meta=meta,
                )
            except Exception as e:
                logger.warning(
                    f"Error converting chunk {chunk.id} to {converter.format}: {e}"
                )
                continue
            new_chunks.append(new_chunk_entity)

        if num_generations and len(new_chunks) >= num_generations:
            break

    return new_chunks


@click.command()
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(),
    required=True,
    help="Sqlite3 file to save the database to",
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
    num_generations: int | None,
    preview: bool,
):
    logger.info(f"Generating input chunks from {db_file}")

    db_session = get_db_session(f"sqlite:///{db_file}")

    try:
        chunks = get_json_chunks_with_schema(db_session)
        random.shuffle(chunks)

        new_chunks = generate_format_converted_chunks(
            chunks,
            num_generations=num_generations,
        )
        logger.info(f"Generated {len(new_chunks)} synthetic chunks")

        deduplicated_chunks = deduplicate_chunks(new_chunks)

        logger.info(f"Deduplicated {len(deduplicated_chunks)} synthetic chunks")

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
