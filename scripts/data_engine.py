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
from any2json.agents.schema_generator import JSONSchemaGeneratorAgent
from any2json.agents.chunk_generator import JSONChunkGeneratorAgent
from any2json.data_engine.generators.converters.utils import (
    generate_synthetic_format_conversions,
)
from any2json.data_engine.helpers import (
    deduplicate_chunks,
    extract_json_chunks,
    generate_chunks_for_schemas,
    generate_schemas_for_chunks,
    get_json_chunks_with_no_schema,
    get_json_chunks_with_schema,
    get_json_documents,
    generate_synthetic_chunks,
    generate_synthetic_schemas,
    map_chunks_to_existing_schemas,
)
from any2json.database.client import db_session_scope
from any2json.database.helpers import get_dangling_schema_ids
from any2json.database.models import JsonSchema
from any2json.infinigram import InfiniGramAPI
from any2json.utils import logger, configure_loggers, stringify_content
from any2json.dataset_processors import get_dataset_processor

from any2json.data_engine.generators.synthetic.pandas_generator import PandasGenerator
from any2json.data_engine.utils import preview_chunks, save_chunks_to_db


PREVIEW = False
DB_FILE = "data/database.db"


@click.group()
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Sqlite3 file to read the database from",
)
@click.option(
    "--preview",
    is_flag=True,
    help="Preview the changes, don't commit to database",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed",
)
def cli(db_file: str, preview: bool, seed: int):
    global PREVIEW
    PREVIEW = preview

    global DB_FILE
    DB_FILE = db_file

    global SEED
    SEED = seed

    random.seed(SEED)

    logger.info(
        f"Using database file: {DB_FILE}, preview mode: {PREVIEW}, seed: {SEED}"
    )


# Section 1: Loading inputs

# Step 1.1: Download datasets


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
        "interstellarninja/json-mode-reasoning": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "interstellarninja/json-mode-verifiable": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "interstellarninja/json-mode-agentic-reasoning": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "interstellarninja/json-schema-store-reasoning": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "dataunitylab/json-schema": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "dataunitylab/json-schema-descriptions": {
            "args": (),
            "kwargs": {"split": "train"},
        },
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


# Step 1.2: Process all source datasets, extract inputs from them


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, dir_okay=True))
@click.option(
    "--num-samples-per-dataset",
    default=None,
    type=int,
    help="Number of documents to load from dataset",
)
def process_dataset(input_dir: str, num_samples_per_dataset: int):
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        logger.info(f"Processing {input_dir}")
        processor = get_dataset_processor(input_dir)
        processor(db_session, input_dir, num_samples_per_dataset)


# Step 1.3: Get json chunks from Infinigram


@cli.command()
@click.option(
    "--num-chunks",
    default=10,
    type=int,
    help="Number of chunks to retrieve",
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
    num_chunks: int,
    infinigram_url: str,
    infinigram_index: str,
    format: str,
):
    logger.info(f"Getting chunks from Infinigram in {format} format")

    infinigram_api = InfiniGramAPI(
        index=infinigram_index,
        max_clause_freq=500000,
        max_diff_tokens=1000,
        timeout=20,
        max_retries=3,
        max_concurrent_requests=10,
    )

    queries = {
        "json": "```json",
        "xml": "```xml",
        "csv": "```csv",
        "yaml": "```yaml",
        "toml": "```toml",
        "html": "```html",
        "html_table": "<table> AND </table>",
    }

    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
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

        new_per_document_chunks = []
        for chunk_list in per_document_chunks:
            chunk_list = [
                chunk
                for chunk in chunk_list
                if len(stringify_content(chunk, format)) >= 100
            ]
            new_per_document_chunks.append(chunk_list)
        per_document_chunks = new_per_document_chunks

        if not document_content:
            logger.info("No results found")
            return

        logger.info(f"Processing API results and saving chunks to database")

        meta = {
            "source": "infinigram",
            "infinigram_url": infinigram_url,
            "infinigram_index": infinigram_index,
        }
        if PREVIEW:
            preview_chunks(document_content, document_meta, per_document_chunks)

        save_chunks_to_db(
            db_session,
            document_content=document_content,
            document_meta=document_meta,
            per_document_chunks=per_document_chunks,
            meta=meta,
            format=format,
        )


# Section 2: Processing inputs

# Step 2.1: Extract json chunks from source documents


@cli.command(
    name="extract-json-chunks",
)
@click.option(
    "--max-depth",
    default=10,
    type=int,
    help="Maximum depth to generate chunks from",
)
@click.option(
    "--frac-per-document",
    default=0.2,
    type=float,
    help="Fraction of chunks to take from each document",
)
@click.option(
    "--max-chunks",
    default=None,
    type=int,
    help="Maximum number of chunks to generate",
)
def extract_json_chunks_command(
    max_depth: int, frac_per_document: float, max_chunks: int | None
):
    logger.info(f"Extracting json chunks from {DB_FILE}")

    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        documents = get_json_documents(db_session)
        chunks = extract_json_chunks(
            documents,
            max_depth=max_depth,
            frac_per_document=frac_per_document,
            max_chunks=max_chunks,
        )
        logger.info(f"Generated {len(chunks)} input chunks")

        deduplicated_chunks = deduplicate_chunks(chunks)

        db_session.add_all(deduplicated_chunks)
        logger.info(f"After deduplication: adding {len(deduplicated_chunks)} chunks")

        if PREVIEW:
            print("Previewing last 10 chunks")
            for chunk in deduplicated_chunks[-10:]:
                print(f"{chunk.parent_document_id=}")
                print(f"{chunk.content=}")
                print()
                print()
                print()


# Section 3: Generating synthetic data


# Step 3.1: Generate synthetic data from pandas


@cli.command()
@click.option(
    "--num-chunks",
    default=10,
    type=int,
    help="Number of chunks to generate",
)
def generate_pandas_chunks(num_chunks: int):
    logger.info(f"Generating input chunks from {DB_FILE}")

    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        generator_class = PandasGenerator

        input_chunks, schemas, output_chunks, schema_conversions = (
            generate_synthetic_chunks(
                db_session,
                num_chunks,
                generator_class,
            )
        )

        if PREVIEW:
            for input_chunk, schema, output_chunk, schema_conversion in zip(
                input_chunks,
                schemas,
                output_chunks,
                schema_conversions,
                strict=True,
            ):
                print(
                    f"{schema_conversion.input_chunk.content_type} ({schema_conversion.meta['generator_state'].get('input_orient')}) -> {schema_conversion.output_chunk.content_type}"
                )
                print(f"Input chunk:\n{input_chunk.content}")
                print()
                print(f"Schema:\n{schema.content}")
                print()
                print(f"Output chunk:\n{output_chunk.content}")
                print()


# Step 3.2 Map chunks to existing schemas via tfidf


@cli.command(
    name="map-chunks",
)
def map_chunks_command():
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        chunks = get_json_chunks_with_no_schema(db_session)
        results = map_chunks_to_existing_schemas(db_session, chunks)

        mapped = 0
        for chunk, schema_id in zip(chunks, results, strict=True):
            if schema_id:
                chunk.schema_id = schema_id
                mapped += 1
        logger.info(f"Mapped {mapped} chunks to existing schemas")

        db_session.add_all(chunks)


# Step 3.3: Generate schemas for json chunks with no schema


@cli.command(
    name="generate-schemas",
)
@click.option(
    "--num-chunks",
    default=None,
    type=int,
    required=False,
)
@click.option(
    "--model-name",
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
    num_chunks: int | None,
    model_name: str,
    enable_thinking: bool,
    max_retries: int,
):
    logger.warning(
        "This command commits after every schema generation and is not easily reversible!"
    )

    api_key = os.getenv("GEMINI_API_KEY")

    assert api_key, "GEMINI_API_KEY is not set"

    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        chunks = get_json_chunks_with_no_schema(db_session)
        random.shuffle(chunks)

        if num_chunks:
            chunks = chunks[:num_chunks]

        logger.info(f"Loaded {len(chunks)} JSON chunks with no schema for processing")

        schema_agent = JSONSchemaGeneratorAgent(
            model_name=model_name,
            max_retries=max_retries,
            enable_thinking=enable_thinking,
        )

        schemas_generated, updated_chunks = asyncio.run(
            generate_schemas_for_chunks(
                db_session=db_session,
                chunks=chunks,
                schema_agent=schema_agent,
            )
        )

        logger.info(
            f"Generated {schemas_generated} schemas, updated {updated_chunks} chunks"
        )


# Step 3.4: Generate chunks for schemas with no chunks


@cli.command(
    name="generate-chunks",
)
@click.option(
    "--num-schemas",
    default=None,
    type=int,
    required=False,
)
@click.option(
    "--model-name",
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
def generate_chunks_command(
    num_schemas: int | None,
    model_name: str,
    enable_thinking: bool,
    max_retries: int,
):
    logger.warning(
        "This command commits after every generation and is not easily reversible!"
    )

    api_key = os.getenv("GEMINI_API_KEY")

    assert api_key, "GEMINI_API_KEY is not set"

    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        schema_ids = get_dangling_schema_ids(db_session)
        schemas = (
            db_session.query(JsonSchema).filter(JsonSchema.id.in_(schema_ids)).all()
        )
        random.shuffle(schemas)

        if num_schemas:
            schemas = schemas[:num_schemas]

        logger.info(f"Loaded {len(schemas)} JSON schemas with no chunks for processing")

        agent = JSONChunkGeneratorAgent(
            model_name=model_name,
            max_retries=max_retries,
            enable_thinking=enable_thinking,
        )

        chunks_generated, updated_schemas, errors = asyncio.run(
            generate_chunks_for_schemas(
                db_session=db_session,
                schemas=schemas,
                agent=agent,
            )
        )

        logger.info(
            f"Generated {chunks_generated} chunks, updated {updated_schemas} schemas, {errors} errors"
        )


# Step 3.5: Generate synthetic schemas from existing schemas


@cli.command(
    name="generate-synthetic-schemas",
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
def generate_synthetic_schemas_command(
    num_variations_per_schema: int,
    num_generations: int | None,
):
    logger.info(f"Generating synthetic schemas from {DB_FILE}")

    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
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

        if PREVIEW:
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


# Step 3.6: Generate synthetic format conversions from JSON to other formats


@cli.command(
    name="generate-synthetic-format-conversions",
)
@click.option(
    "--num-generations",
    default=None,
    type=int,
    required=False,
)
def generate_synthetic_format_conversions_command(
    num_generations: int | None,
):
    logger.info(f"Generating synthetic format conversions from {DB_FILE}")

    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        synthetic_chunks, schema_conversions = generate_synthetic_format_conversions(
            db_session,
            num_generations=num_generations,
        )

        if PREVIEW:
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


if __name__ == "__main__":
    load_dotenv(override=True)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    cli()
