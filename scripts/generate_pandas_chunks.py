import logging
import click
import json

from tqdm import tqdm

from any2json.database.client import get_db_session
from any2json.database.models import Chunk, JsonSchema
from any2json.enums import ContentType
from any2json.data_engine.generators.synthetic.pandas_generator import PandasGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.command()
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
def run(db_file: str, num_chunks: int, preview: bool):
    logger.info(f"Generating input chunks from {db_file}")

    db_session = get_db_session(f"sqlite:///{db_file}")

    try:
        input_chunks = []
        schemas = []
        output_chunks = []
        for i in tqdm(range(num_chunks)):
            generator = PandasGenerator()
            generator.setup()
            generator_state = generator.get_state()
            input_format = ContentType(generator.format_name.upper()).value
            input_str, schema, output_json = generator.generate_triplet()

            meta = {
                "generator": "PandasGenerator",
                "generator_state": generator_state,
            }

            schema_entity = JsonSchema(
                content=schema,
                is_synthetic=True,
                meta=meta,
            )

            chunk_entity = Chunk(
                content=input_str,
                content_type=input_format,
                schema=schema_entity,
                meta=meta,
                is_synthetic=True,
            )

            output_chunk_entity = Chunk(
                content=output_json,
                content_type=ContentType.JSON.value,
                schema=schema_entity,
                meta=meta,
                is_synthetic=True,
            )

            input_chunks.append(chunk_entity)
            schemas.append(schema_entity)
            output_chunks.append(output_chunk_entity)

            db_session.add(schema_entity)
            db_session.add(chunk_entity)
            db_session.add(output_chunk_entity)

        logger.info(f"Generated {len(input_chunks)} input chunks")

        if not preview:
            db_session.commit()
        else:
            for input_chunk, schema, output_chunk in zip(
                input_chunks,
                schemas,
                output_chunks,
                strict=True,
            ):
                print(f"{input_chunk.content=}")
                print()
                print(f"{schema.content=}")
                print()
                print(f"{output_chunk.content=}")

    except Exception as e:
        db_session.rollback()
        raise e


if __name__ == "__main__":
    run()
