import logging
import click
import os
import json
from datasets import Dataset
from dataclasses import asdict

from sqlalchemy.orm import Session

from any2json.data_engine.utils import (
    deduplicate_chunks,
    get_chunks_from_record,
)
from any2json.database.client import get_db_session
from any2json.database.models import Chunk, SourceDocument
from any2json.enums import ContentType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_json_documents(db_session: Session) -> list[SourceDocument]:
    return (
        db_session.query(SourceDocument)
        .filter(SourceDocument.content_type == ContentType.JSON.value)
        .all()
    )


def generate_json_chunks(
    documents: list[SourceDocument], max_depth: int = 3
) -> list[Chunk]:
    chunks = []
    for document in documents:
        json_content = json.loads(document.content)

        chunk_jsons = get_chunks_from_record(json_content, max_depth=max_depth)

        for chunk_json in chunk_jsons:
            chunks.append(
                Chunk(
                    parent_document_id=document.id,
                    content=json.dumps(chunk_json),
                    content_type=ContentType.JSON.value,
                    is_synthetic=False,
                )
            )

    return chunks


@click.command()
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
def run(db_file: str, max_depth: int):
    logger.info(f"Generating input chunks from {db_file}")

    db_session = get_db_session(f"sqlite:///{db_file}")

    try:
        documents = get_json_documents(db_session)
        chunks = generate_json_chunks(documents, max_depth=max_depth)
        logger.info(f"Generated {len(chunks)} input chunks")

        deduplicated_chunks = deduplicate_chunks(chunks)
        logger.info(f"Deduplicated {len(deduplicated_chunks)} input chunks")

        db_session.add_all(deduplicated_chunks)
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        raise e
    finally:
        db_session.close()


if __name__ == "__main__":
    run()
