from dataclasses import asdict, dataclass
from typing import Any
import click
import logging
import json
import os
from tqdm.auto import tqdm
import instructor
from any2json.containers import ChunkWithSchema, InputJSONChunk
from any2json.data_engine.agents import (
    JSONSchemaValidationAgent,
    SchemaAgentInputSchema,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_schemas_for_chunks(
    chunks: list[InputJSONChunk],
    schema_agent: JSONSchemaValidationAgent,
) -> list[ChunkWithSchema]:
    chunks_with_schemas = []
    for chunk in tqdm(chunks):
        schema = generate_schema_for_chunk(
            chunk,
            schema_agent,
        )
        if schema:
            chunks_with_schemas.append(
                ChunkWithSchema(
                    id=chunk.id,
                    data=chunk.data,
                    source_dataset_name=chunk.source_dataset_name,
                    source_dataset_index=chunk.source_dataset_index,
                    schema=schema,
                )
            )

    return chunks_with_schemas


def generate_schema_for_chunk(
    chunk: InputJSONChunk,
    schema_agent: JSONSchemaValidationAgent,
) -> dict:
    original_data = chunk.data

    input_string = json.dumps(original_data, indent=1)

    try:
        schema = schema_agent.generate_and_validate_schema(
            SchemaAgentInputSchema(input_string=input_string)
        )
    except Exception as e:
        logger.error(f"Failed to generate schema for chunk {chunk}")
        logger.error(e)
        return None

    return schema


def create_schema_agent(
    model: str = "gemini-2.5-flash-preview-05-20",
    max_retries: int = 1,
    **kwargs,
) -> JSONSchemaValidationAgent:
    client = instructor.from_provider(
        f"google/{model}",
        **kwargs,
    )
    return JSONSchemaValidationAgent(client, model, max_retries)


def load_chunks(input_dir: str) -> list[InputJSONChunk]:
    with open(f"{input_dir}/input_json_chunks.json", "r") as f:
        chunks = json.load(f)
    return [InputJSONChunk(**chunk) for chunk in chunks]


@click.command()
@click.option(
    "--input-dir",
    default="data/intermediate",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output-dir",
    default="data/intermediate",
    type=click.Path(),
    required=True,
)
@click.option(
    "--model",
    default="gemini-2.0-flash",
    type=str,
    required=True,
)
@click.option(
    "--num-chunks",
    default=None,
    type=int,
    required=False,
)
def run(
    input_dir: str,
    output_dir: str,
    model: str,
    num_chunks: int,
):
    logger.info(f"Generating schemas from chunks in {input_dir}")

    api_key = os.getenv("GEMINI_API_KEY")

    schema_agent = create_schema_agent(
        model=model,
        max_retries=3,
    )

    chunks = load_chunks(input_dir)

    if num_chunks:
        chunks = chunks[:num_chunks]

    logger.info(f"Loaded {len(chunks)} chunks")

    chunks_with_schemas = generate_schemas_for_chunks(
        chunks,
        schema_agent,
    )

    logger.info(f"Saving chunks with schemas to {output_dir}")

    chunk_dicts = [asdict(chunk) for chunk in chunks_with_schemas]
    with open(f"{output_dir}/chunks_with_schemas.json", "w") as f:
        json.dump(chunk_dicts, f, indent=2)


if __name__ == "__main__":
    run()
