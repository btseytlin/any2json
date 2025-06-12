from dataclasses import asdict
import random
import click
import logging
import json

from sqlalchemy.orm import Session
from any2json.database.client import get_db_session
from any2json.database.models import Chunk, TrainingSample
from any2json.enums import ContentType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_format_conversion_samples(
    db_session: Session, num_samples: int | None
) -> list[TrainingSample]:
    """Find all samples that can be made by connecting chunks in different formats that have the same schema"""

    samples = []

    non_json_chunks_with_schema = (
        db_session.query(Chunk)
        .filter(
            Chunk.schema_id.isnot(None), Chunk.content_type != ContentType.JSON.value
        )
        .all()
    )

    for chunk in non_json_chunks_with_schema:
        json_chunk = (
            db_session.query(Chunk)
            .filter(
                Chunk.schema_id == chunk.schema_id,
                Chunk.content_type == ContentType.JSON.value,
            )
            .first()
        )

        samples.append(
            TrainingSample(
                input_chunk=chunk,
                target_schema=chunk.schema,
                output_chunk=json_chunk,
                meta={
                    "sample_type": "non_json_to_json",
                },
            )
        )

        if len(samples) >= num_samples:
            break

    return samples


def generate_json_to_json_samples(
    db_session: Session, num_samples: int | None
) -> list[TrainingSample]:
    """Find all samples that can be made by connecting json chunks that have the same schema"""

    samples = []

    json_chunks_with_schema = (
        db_session.query(Chunk)
        .filter(
            Chunk.schema_id.isnot(None), Chunk.content_type == ContentType.JSON.value
        )
        .all()
    )


@click.command()
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(),
    required=True,
    help="Sqlite3 file to save the database to",
)
@click.option(
    "--num-samples",
    default=None,
    type=int,
    required=True,
)
@click.option(
    "--preview",
    is_flag=True,
    help="Preview the generated chunks, don't save to database",
)
def run(
    db_file: str,
    num_samples: int | None,
    preview: bool,
):
    logger.info(f"Generating samples from {db_file}")

    db_session = get_db_session(f"sqlite:///{db_file}")

    try:
        format_samples = generate_format_conversion_samples(db_session, num_samples)

        logger.info(
            f"Generated {len(format_samples)} samples for different formats to JSON conversions"
        )

        # json_to_json_samples = generate_json_to_json_samples(db_session, num_samples)

    except Exception as e:
        db_session.rollback()
        raise e
    finally:
        db_session.close()


if __name__ == "__main__":
    run()
