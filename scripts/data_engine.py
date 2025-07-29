import asyncio
import json
import os
import random
import click
from dotenv import load_dotenv
import os
import time
import click
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from any2json.data_engine.agents import JSONSchemaValidationAgent
from any2json.data_engine.generators.converters.utils import (
    generate_synthetic_format_conversions,
)
from any2json.data_engine.helpers import (
    deduplicate_chunks,
    extract_json_chunks,
    generate_schemas_for_chunks,
    get_json_chunks_with_no_schema,
    get_json_chunks_with_schema,
    get_json_documents,
    generate_synthetic_chunks,
    generate_synthetic_schemas,
)
from any2json.database.client import db_session_scope
from any2json.infinigram import InfiniGramAPI
from any2json.utils import logger, configure_loggers
from any2json.dataset_processors import get_dataset_processor

from any2json.data_engine.generators.synthetic.pandas_generator import PandasGenerator
from any2json.data_engine.utils import preview_chunks, save_chunks_to_db


@click.group()
def cli():
    pass


# Step 0: Download datasets


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


# Step 1: Process all source datasets, extract inputs from them


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


# Step 2: Extract json chunks from source documents


@cli.command(
    name="extract-json-chunks",
)
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(),
    required=True,
    help="Sqlite3 file to save the database to",
)
@click.option(
    "--max-depth",
    default=3,
    type=int,
    help="Maximum depth to generate chunks from",
)
def extract_json_chunks_command(db_file: str, max_depth: int):
    logger.info(f"Extracting json chunks from {db_file}")

    with db_session_scope(f"sqlite:///{db_file}") as db_session:
        documents = get_json_documents(db_session)
        chunks = extract_json_chunks(documents, max_depth=max_depth)
        logger.info(f"Generated {len(chunks)} input chunks")

        deduplicated_chunks = deduplicate_chunks(chunks)

        db_session.add_all(deduplicated_chunks)
        logger.info(f"After deduplication: adding {len(deduplicated_chunks)} chunks")


# Step 2.5: Get json chunks from Infinigram


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
    help="Number of chunks to retrieve",
)
@click.option(
    "--preview",
    is_flag=True,
    help="Preview the generated chunks, don't save to database",
)
@click.option(
    "--infinigram-url",
    default="https://api.infini-gram.io/",
    type=str,
    help="URL of the Infogram API",
)
@click.option(
    "--infinigram-index",
    default="v4_olmo-2-0325-32b-instruct_llama",
    type=str,
)
@click.option(
    "--format",
    default="json",
    type=str,
    help="Format of the chunks to retrieve",
)
def get_from_infinigram(
    db_file: str,
    num_chunks: int,
    preview: bool,
    infinigram_url: str,
    infinigram_index: str,
    format: str,
):
    logger.info(f"Getting chunks from Infinigram in {format} format")

    infinigram_api = InfiniGramAPI(
        index=infinigram_index,
        max_clause_freq=500000,
        max_diff_tokens=1000,
        timeout=5,
        max_retries=3,
        max_concurrent_requests=25,
    )

    queries = {
        "json": "```json",
        "xml": "```xml",
        "csv": "```csv",
        "yaml": "```yaml",
        "toml": "```toml",
        "html": "```html",
    }

    with db_session_scope(f"sqlite:///{db_file}") as db_session:
        query = queries[format]

        find_results = asyncio.run(infinigram_api.find_documents(query))

        if not find_results:
            logger.info("No documents found matching the query.")
            return

        document_content, document_meta, per_document_chunks = (
            infinigram_api.fetch_and_process_documents(
                num_chunks=num_chunks,
                segment_by_shard=find_results["segment_by_shard"],
                format=format,
                query=query,
            )
        )

        if not document_content:
            logger.info("No results found")
            return

        logger.info(f"Processing API results and saving chunks to database")

        meta = {
            "source": "infinigram",
            "infinigram_url": infinigram_url,
            "infinigram_index": infinigram_index,
        }
        if preview:
            preview_chunks(document_content, document_meta, per_document_chunks)
            raise Exception("Preview mode, not saving to database")

        save_chunks_to_db(
            db_session,
            document_content=document_content,
            document_meta=document_meta,
            per_document_chunks=per_document_chunks,
            meta=meta,
            format=format,
        )


# Step 3: Generate synthetic data from pandas


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
        generator_class = PandasGenerator

        input_chunks, schemas, output_chunks, schema_conversions = (
            generate_synthetic_chunks(
                db_session,
                num_chunks,
                generator_class,
            )
        )

        if preview:
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

            raise Exception("Preview mode, not saving to database")


# Step 4: Generate schemas for json chunks with no schema


@cli.command(
    name="generate-schemas",
)
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(),
    required=True,
    help="Sqlite3 file to save the database to",
)
@click.option(
    "--num-chunks",
    default=None,
    type=int,
    required=False,
)
@click.option(
    "--model",
    default="gemini-2.5-flash-lite",
    type=str,
    required=True,
)
@click.option(
    "--enable-thinking",
    default=True,
    type=bool,
    required=False,
)
@click.option(
    "--max-retries",
    default=3,
    type=int,
    required=False,
)
def generate_schemas_command(
    db_file: str,
    num_chunks: int | None,
    model: str,
    max_retries: int,
    enable_thinking: bool,
):
    logger.info(f"Generating synthetic schemas from {db_file}")
    logger.warning(
        "This command commits after every schema generation and is not easily reversible!"
    )

    api_key = os.getenv("GEMINI_API_KEY")

    assert api_key, "GEMINI_API_KEY is not set"

    with db_session_scope(f"sqlite:///{db_file}") as db_session:
        chunks = get_json_chunks_with_no_schema(db_session)
        random.shuffle(chunks)

        if num_chunks:
            chunks = chunks[:num_chunks]

        logger.info(f"Loaded {len(chunks)} JSON chunks with no schema for processing")

        schema_agent = JSONSchemaValidationAgent(model, max_retries, enable_thinking)

        schemas, chunks = generate_schemas_for_chunks(
            db_session=db_session,
            chunks=chunks,
            schema_agent=schema_agent,
        )

        logger.info(f"Generated {len(schemas)} schemas for {len(chunks)} chunks")


# Step 5: Generate synthetic schemas from existing schemas


@cli.command(
    name="generate-synthetic-schemas",
)
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
def generate_synthetic_schemas_command(
    db_file: str,
    num_variations_per_schema: int,
    num_generations: int | None,
    preview: bool,
):
    logger.info(f"Generating synthetic schemas from {db_file}")

    with db_session_scope(f"sqlite:///{db_file}") as db_session:
        chunks = get_json_chunks_with_schema(db_session)
        random.shuffle(chunks)

        new_schemas, new_chunks, new_schema_conversions = generate_synthetic_schemas(
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
        db_session.add_all(new_schema_conversions)

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
            raise Exception("Preview mode, not saving to database")


# Step 6: Generate synthetic format conversions from JSON to other formats


@cli.command(
    name="generate-synthetic-format-conversions",
)
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
def generate_synthetic_format_conversions_command(
    db_file: str,
    num_generations: int | None,
    preview: bool,
):
    logger.info(f"Generating synthetic format conversions from {db_file}")

    with db_session_scope(f"sqlite:///{db_file}") as db_session:
        synthetic_chunks, schema_conversions = generate_synthetic_format_conversions(
            db_session,
            num_generations=num_generations,
        )

        if preview:
            for synthetic_chunk, schema_conversion in zip(
                synthetic_chunks,
                schema_conversions,
                strict=True,
            ):
                print(f"{schema_conversion.input_chunk.content=}")
                print()
                print(f"{schema_conversion.schema.content=}")
                print()
                print(f"{schema_conversion.output_chunk.content=}")
                print()
                print()

            raise Exception("Preview mode, not saving to database")


if __name__ == "__main__":
    load_dotenv(override=True)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    cli()
