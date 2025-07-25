import os
import click
from dotenv import load_dotenv
import logging
import os
import time
import click
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from any2json.database.client import db_session_scope
from any2json.utils import logger, configure_loggers
from any2json.dataset_processors import get_dataset_processor

import logging
import click
import json

from tqdm import tqdm

from any2json.database.client import get_db_session
from any2json.database.models import Chunk, JsonSchema, SchemaConversion
from any2json.enums import ContentType
from any2json.data_engine.generators.synthetic.pandas_generator import PandasGenerator


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--output-dir",
    default="data/raw",
    type=click.Path(exists=True),
    required=True,
    help="Directory to save the inputs",
)
@click.option(
    "--max-records",
    default=None,
    type=int,
    help="Maximum number of records to download from each dataset (for partial downloads)",
)
@click.option(
    "--overwrite",
    default=False,
    type=bool,
    help="Overwrite existing datasets",
)
def download_datasets(output_dir, max_records, overwrite):
    logger.info(f"Downloading datasets to {output_dir}")
    hf_token = os.getenv("HF_TOKEN")

    datasets = {
        "ChristianAzinn/json-training": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "wikimedia/structured-wikipedia": {
            "args": ("20240916.en",),
            "kwargs": {"split": "train"},
        },
        # "interstellarninja/json-mode-reasoning": {
        #     "args": (),
        #     "kwargs": {"split": "train"},
        # },
        # "interstellarninja/json-mode-verifiable": {
        #     "args": (),
        #     "kwargs": {"split": "train"},
        # },
        # "interstellarninja/json-mode-agentic-reasoning": {
        #     "args": (),
        #     "kwargs": {"split": "train"},
        # },
        # "interstellarninja/json-schema-store-reasoning": {
        #     "args": (),
        #     "kwargs": {"split": "train"},
        # },
        # "mdhasnainali/job-html-to-json": {
        #     "args": (),
        #     "kwargs": {"split": "train"},
        # },
        # "shubh303/Invoice-to-Json": {
        #     "args": (),
        #     "kwargs": {"split": "train"},
        # },
        # "GokulRajaR/invoice-ocr-json": {
        #     "args": (),
        #     "kwargs": {"split": "train"},
        # },
        # "dataunitylab/json-schema": {
        #     "args": (),
        #     "kwargs": {"split": "train"},
        # },
        # "dataunitylab/json-schema-keywords": {
        #     "args": (),
        #     "kwargs": {"split": "train"},
        # },
        # "dataunitylab/json-schema-descriptions": {
        #     "args": (),
        #     "kwargs": {"split": "train"},
        # },
    }

    for dataset_id, dataset_info in datasets.items():
        args, kwargs = dataset_info["args"], dataset_info["kwargs"]
        dataset_org, dataset_name = dataset_id.split("/")
        output_path = f"{output_dir}/{dataset_org}/{dataset_name}"
        logger.info(f"Downloading dataset {dataset_org}/{dataset_id} to {output_path}")
        if os.path.exists(output_path) and not overwrite:
            logger.info("Already exists, skipping")
            continue

        try:
            if max_records:
                dataset = load_dataset(
                    dataset_id,
                    *args,
                    **kwargs,
                    token=hf_token,
                    streaming=True,
                )
                limited_dataset = dataset.take(max_records)

                dataset_to_save = Dataset.from_generator(lambda: limited_dataset)
            else:
                dataset_to_save = load_dataset(
                    dataset_id, *args, **kwargs, token=hf_token
                )

            logger.info(f"Saving dataset {dataset_id} to output_path")
            dataset_to_save.save_to_disk(output_path)
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_id}: {e}")
        time.sleep(3)


# Step 1: Process all source datasets, extract whatever we can from them


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, dir_okay=True))
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(exists=True, dir_okay=False, writable=True),
    required=True,
    help="Sqlite3 file to save the database to",
)
@click.option(
    "--num-samples-per-dataset",
    default=None,
    type=int,
    help="Number of documents to load from dataset",
)
def process_dataset(input_dir: str, db_file: str, num_samples_per_dataset: int):
    with db_session_scope(f"sqlite:///{db_file}") as db_session:
        logger.info(f"Processing {input_dir}")
        processor = get_dataset_processor(input_dir)
        processor(db_session, input_dir, num_samples_per_dataset)


@cli.command()
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(),
    required=True,
    help="Sqlite3 file to save the database to",
)
@click.option(
    "--num-chunks",
    default=10,
    type=int,
    help="Number of chunks to generate",
)
@click.option(
    "--preview",
    is_flag=True,
    help="Preview the generated chunks, don't save to database",
)
def generate_pandas_chunks(db_file: str, num_chunks: int, preview: bool):
    logger.info(f"Generating input chunks from {db_file}")

    with db_session_scope(f"sqlite:///{db_file}") as db_session:
        input_chunks = []
        schemas = []
        output_chunks = []
        schema_conversions = []
        for i in tqdm(range(num_chunks)):
            generator = PandasGenerator()
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

            for input_chunk, schema, output_chunk, schema_conversion in zip(
                input_chunks,
                schemas,
                output_chunks,
                schema_conversions,
                strict=True,
            ):
                print(f"{input_chunk.content=}")
                print()
                print(f"{schema.content=}")
                print()
                print(f"{output_chunk.content=}")
                print()
                print(f"{schema_conversion=}")
                print()
                print()

            if preview:
                raise Exception("Preview mode, not saving to database")


if __name__ == "__main__":
    load_dotenv(override=True)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "INFO"),
    )
    cli()
