from dataclasses import asdict
import random
import click
import logging
import json
import os

import instructor
from any2json.containers import ChunkWithSchema, InputJSONChunk, Sample
from any2json.data_engine.agents import (
    JSONSchemaValidationAgent,
    SchemaAgentInputSchema,
)
from any2json.data_engine.generators import (
    SampleGenerator,
    ToPythonStringGenerator,
    VarySchemaSampleGenerator,
    ToCsvGenerator,
    ToHtmlTableGenerator,
    ToHtmlTreeGenerator,
    ToXmlGenerator,
    ToYamlGenerator,
    ToTomlGenerator,
    ToMarkdownTableGenerator,
    ToPlainTextGenerator,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_samples_from_chunks(
    chunks: list[ChunkWithSchema],
    sample_generators: list[SampleGenerator],
    num_samples_from_chunk: int,
    num_chunks: int,
) -> list[Sample]:
    samples = []
    for chunk in chunks[:num_chunks]:
        generator = random.choice(sample_generators)
        chunk_samples = generate_samples_from_chunk(
            chunk,
            generator,
            num_samples_from_chunk,
        )
        for sample in chunk_samples:
            sample.chunk_id = chunk.id

        samples.extend(chunk_samples)

    return samples


def generate_samples_from_chunk(
    chunk: ChunkWithSchema,
    sample_generator: SampleGenerator,
    num_samples: int,
) -> list[Sample]:
    original_data = chunk.data

    schema = chunk.schema

    samples = []
    for _ in range(num_samples):
        samples.append(sample_generator.generate_sample(original_data, schema))

    return samples


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


def load_chunks(input_dir: str) -> list[ChunkWithSchema]:
    with open(f"{input_dir}/chunks_with_schemas.json", "r") as f:
        chunks = json.load(f)
    return [ChunkWithSchema(**chunk) for chunk in chunks]


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
    "--num-samples-from-chunk",
    default=3,
    type=int,
    required=True,
)
@click.option(
    "--num-chunks",
    default=2,
    type=int,
    required=True,
)
def run(
    input_dir: str,
    output_dir: str,
    num_samples_from_chunk: int,
    num_chunks: int,
):
    logger.info(f"Generating samples from chunks in {input_dir}")

    sample_generators = [
        VarySchemaSampleGenerator(),
        # ToCsvGenerator(),
        # ToHtmlTableGenerator(),
        # ToHtmlTreeGenerator(),
        ToPythonStringGenerator(),
        # ToXmlGenerator(),
        ToYamlGenerator(),
        ToTomlGenerator(),
        ToMarkdownTableGenerator(),
        ToPlainTextGenerator(),
    ]

    chunks = load_chunks(input_dir)

    logger.info(f"Loaded {len(chunks)} chunks")

    samples = generate_samples_from_chunks(
        chunks,
        sample_generators,
        num_samples_from_chunk,
        num_chunks,
    )

    logger.info(f"Generated {len(samples)} samples")

    logger.info(f"Saving samples to {output_dir}")

    sample_dicts = [asdict(sample) for sample in samples]
    with open(f"{output_dir}/samples.json", "w") as f:
        json.dump(sample_dicts, f, indent=2)


if __name__ == "__main__":
    run()
