import logging
import click
import json

from tqdm import tqdm

from any2json.database.client import get_db_session
from any2json.database.models import Chunk, JsonSchema, SchemaConversion
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

        if not preview:
            db_session.commit()
        else:
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

    except Exception as e:
        db_session.rollback()
        raise e
    finally:
        db_session.close()


if __name__ == "__main__":
    run()
