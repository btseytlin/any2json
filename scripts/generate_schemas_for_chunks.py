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
                    f"Error: {e}\nFailed to commit chunk {chunk.content} schema: {schema_entity.content}",
                    exc_info=True,
                )
                raise e

    return schemas, updated_chunks


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
    except Exception as e:
        logger.error(f"Failed to generate schema for JSON {json_content}")
        logger.error(e)
        return None

    return schema


def create_schema_agent(
    model: str = "gemini-2.0-flash",
    max_retries: int = 1,
    **kwargs,
) -> JSONSchemaValidationAgent:
    client = instructor.from_provider(
        f"google/{model}",
        **kwargs,
    )
    return JSONSchemaValidationAgent(client, model, max_retries)


@click.command()
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(),
    required=True,
    help="Sqlite3 file to save the database to",
)
@click.option(
    "--model",
    default="gemini-2.0-flash",
    type=str,
    required=True,
)
@click.option(
    "--max-retries",
    default=2,
    type=int,
    required=False,
    help="Maximum number of retries for schema generation",
)
@click.option(
    "--num-chunks",
    default=None,
    type=int,
    required=False,
)
def run(
    db_file: str,
    model: str,
    max_retries: int,
    num_chunks: int,
):
    logger.info(f"Generating schemas from chunks in {db_file}")

    api_key = os.getenv("GEMINI_API_KEY")

    schema_agent = create_schema_agent(
        model=model,
        max_retries=max_retries,
        api_key=api_key,
    )

    db_session = get_db_session(f"sqlite:///{db_file}")

    try:
        chunks = get_json_chunks_with_no_schema(db_session)
        random.shuffle(chunks)

        if num_chunks:
            chunks = chunks[:num_chunks]

        logger.info(f"Loaded {len(chunks)} JSON chunks with no schema for processing")

        schemas, chunks = generate_schemas_for_chunks(
            db_session=db_session,
            chunks=chunks,
            schema_agent=schema_agent,
        )

        logger.info(f"Generated {len(schemas)} schemas for {len(chunks)} chunks")
    except Exception as e:
        db_session.rollback()
        raise e
    finally:
        db_session.close()


if __name__ == "__main__":
    run()
