from collections import defaultdict
import json
import os
from dotenv import load_dotenv
import click
from sqlalchemy import String, cast, create_engine, func, select
from sqlalchemy.orm import Session

from any2json.database.client import create_tables, get_db_session
from any2json.database.models import Chunk, JsonSchema, SourceDocument
from any2json.utils import configure_loggers, logger


@click.group()
def cli():
    pass


def get_duplicated_documents_by_metadata(
    db_session: Session,
) -> dict[str, list[SourceDocument]]:
    """Return ids of documents that have the same metadata as other documents."""
    duplicate_meta_subquery = (
        select(cast(SourceDocument.meta, String))
        .where(SourceDocument.meta != None)
        .group_by(cast(SourceDocument.meta, String))
        .having(func.count(cast(SourceDocument.meta, String)) > 1)
    )
    query = (
        select(SourceDocument)
        .where(SourceDocument.meta != None)
        .where(cast(SourceDocument.meta, String).in_(duplicate_meta_subquery))
    )
    docs = db_session.execute(query).scalars().all()
    duplicates = defaultdict(list)
    for doc in docs:
        metadata_fingerprint = json.dumps(doc.meta, sort_keys=True)
        duplicates[metadata_fingerprint].append(doc)
    return duplicates


def get_duplicated_documents_by_content(
    db_session: Session,
) -> dict[str, list[SourceDocument]]:
    """Return ids of documents that have the same content as other documents."""
    duplicate_content_subquery = (
        select(SourceDocument.content)
        .where(SourceDocument.content != None)
        .group_by(SourceDocument.content)
        .having(func.count(SourceDocument.content) > 1)
    )
    query = (
        select(SourceDocument)
        .where(SourceDocument.content != None)
        .where(SourceDocument.content.in_(duplicate_content_subquery))
    )
    docs = db_session.execute(query).scalars().all()
    duplicates = defaultdict(list)
    for doc in docs:
        content_fingerprint = doc.content.strip().lower()
        duplicates[content_fingerprint].append(doc)
    return duplicates


def drop_duplicated_documents(
    db_session: Session,
    duplicate_documents: dict[str, list[SourceDocument]],
):
    if not duplicate_documents:
        return

    documents_to_delete = set()
    for _, documents in duplicate_documents.items():
        for document in documents[1:]:
            documents_to_delete.add(document)

    chunks_to_delete = set()
    for doc in documents_to_delete:
        for chunk in doc.chunks:
            chunks_to_delete.add(chunk)

    logger.info(
        f"Found {len(documents_to_delete)} duplicate documents to delete: {[d.id for d in documents_to_delete]}"
    )
    logger.info(
        f"{len(chunks_to_delete)} child chunks will be deleted: {[c.id for c in chunks_to_delete]}"
    )

    for doc in documents_to_delete:
        for chunk in doc.chunks:
            db_session.delete(chunk)
        db_session.delete(doc)


def get_duplicated_chunks(db_session: Session) -> dict[str, list[Chunk]]:
    """Return chunks that have the same content as other chunks.
    Return a dict of content -> list of chunks.
    """
    query = select(Chunk).group_by(Chunk.content).having(func.count(Chunk.content) > 1)
    chunks = db_session.execute(query).scalars().all()
    duplicates = defaultdict(list)
    for chunk in chunks:
        duplicates[chunk.content].append(chunk)
    return duplicates


@cli.command()
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
    help="If true doesnt save changes to the database",
)
def drop_duplicate_documents(db_file: str, preview: bool):
    db_session = get_db_session(f"sqlite:///{db_file}")
    logger.info("Getting duplicate documents by metadata")
    duplicate_documents_by_metadata = get_duplicated_documents_by_metadata(db_session)
    if duplicate_documents_by_metadata:
        logger.info("Dropping duplicate documents by metadata")
        drop_duplicated_documents(db_session, duplicate_documents_by_metadata)
    else:
        logger.info("No duplicate documents by metadata found")

    logger.info("Getting duplicate documents by content")
    duplicate_documents_by_content = get_duplicated_documents_by_content(db_session)
    if duplicate_documents_by_content:
        logger.info("Dropping duplicate documents by content")
        drop_duplicated_documents(db_session, duplicate_documents_by_content)
    else:
        logger.info("No duplicate documents by content found")

    if not preview:
        db_session.commit()
        logger.info("Committed changes to the database")
    else:
        logger.info("Preview mode, not deleting anything")


def get_duplicated_chunks_by_content(
    db_session: Session,
) -> dict[str, list[Chunk]]:
    """Return ids of documents that have the same content as other documents."""
    duplicate_content_subquery = (
        select(Chunk.content)
        .group_by(Chunk.content)
        .having(func.count(Chunk.content) > 1)
    )
    query = select(Chunk).where(Chunk.content.in_(duplicate_content_subquery))
    chunks = db_session.execute(query).scalars().all()
    duplicates = defaultdict(list)
    for chunk in chunks:
        content_fingerprint = chunk.content.strip().lower()
        duplicates[content_fingerprint].append(chunk)
    return duplicates


def drop_duplicated_chunks(
    db_session: Session,
    duplicate_chunks: dict[str, list[Chunk]],
):
    if not duplicate_chunks:
        return

    chunks_to_delete = set()
    for _, chunks in duplicate_chunks.items():
        for chunk in chunks[1:]:
            chunks_to_delete.add(chunk)

    logger.info(
        f"Found {len(chunks_to_delete)} duplicate chunks to delete: {[c.id for c in chunks_to_delete]}"
    )

    for chunk in chunks_to_delete:
        db_session.delete(chunk)


def get_duplicated_schemas_by_content(
    db_session: Session,
) -> dict[str, list[JsonSchema]]:
    """Return schemas that have the same content as other schemas."""
    duplicate_content_subquery = (
        select(JsonSchema.content)
        .group_by(JsonSchema.content)
        .having(func.count(JsonSchema.content) > 1)
    )
    query = select(JsonSchema).where(JsonSchema.content.in_(duplicate_content_subquery))
    schemas = db_session.execute(query).scalars().all()
    duplicates = defaultdict(list)
    for schema in schemas:
        content_fingerprint = json.dumps(schema.content)
        duplicates[content_fingerprint].append(schema)
    return duplicates


def drop_duplicated_schemas(
    db_session: Session,
    duplicate_schemas: dict[str, list[JsonSchema]],
):
    if not duplicate_schemas:
        return

    schemas_to_delete = set()
    for _, schemas in duplicate_schemas.items():
        original_schema = schemas[0]
        for schema in schemas[1:]:
            for chunk in schema.chunks:
                logger.debug(
                    f"Chunk {chunk.id} schema reassigned {chunk.schema_id} -> {original_schema.id}"
                )
                chunk.schema = original_schema
                db_session.add(chunk)
            schemas_to_delete.add(schema)

    logger.info(
        f"Found {len(schemas_to_delete)} duplicate schemas to delete: {[s.id for s in schemas_to_delete]}"
    )

    for schema in schemas_to_delete:
        db_session.delete(schema)


@cli.command()
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(exists=False, dir_okay=False),
    required=True,
    help="Sqlite3 file to read the database from",
)
def init_db(db_file: str):
    db_session = get_db_session(f"sqlite:///{db_file}")
    create_tables(db_session)
    db_session.commit()
    db_session.close()


@cli.command()
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
    help="If true doesnt save changes to the database",
)
def drop_duplicate_chunks(db_file: str, preview: bool):
    db_session = get_db_session(f"sqlite:///{db_file}")
    duplicate_chunks_by_content = get_duplicated_chunks_by_content(db_session)
    if duplicate_chunks_by_content:
        logger.info("Dropping duplicate chunks")
        drop_duplicated_chunks(db_session, duplicate_chunks_by_content)
    else:
        logger.info("No duplicate chunks found")

    if not preview:
        db_session.commit()
        logger.info("Committed changes to the database")
    else:
        logger.info("Preview mode, not deleting anything")


@cli.command()
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
    help="If true doesnt save changes to the database",
)
def drop_duplicate_schemas(db_file: str, preview: bool):
    db_session = get_db_session(f"sqlite:///{db_file}")
    duplicate_schemas_by_content = get_duplicated_schemas_by_content(db_session)
    if duplicate_schemas_by_content:
        logger.info("Dropping duplicate schemas")
        drop_duplicated_schemas(db_session, duplicate_schemas_by_content)
    else:
        logger.info("No duplicate schemas found")

    if not preview:
        db_session.commit()
        logger.info("Committed changes to the database")
    else:
        logger.info("Preview mode, not deleting anything")


if __name__ == "__main__":
    load_dotenv(override=True)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    logger.info("Starting script")
    cli()
