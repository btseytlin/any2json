from collections import defaultdict
import copy
import json
import os
from dotenv import load_dotenv
import click
from sqlalchemy import (
    String,
    cast,
    create_engine,
    delete,
    distinct,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.orm import Session

from any2json.database.client import create_tables, db_session_scope, get_db_session
from any2json.database.helpers import get_dangling_schema_ids
from any2json.database.models import Chunk, JsonSchema, SourceDocument
from any2json.utils import configure_loggers, logger


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
def cli(db_file: str, preview: bool):
    global PREVIEW
    PREVIEW = preview

    global DB_FILE
    DB_FILE = db_file

    logger.info(f"Using database file: {DB_FILE}, preview mode: {PREVIEW}")


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
        doc_metadata = copy.deepcopy(doc.meta or {})
        doc_metadata["source"] = doc.source
        metadata_fingerprint = json.dumps(doc_metadata, sort_keys=True)
        duplicates[metadata_fingerprint].append(doc)
    return duplicates


def get_duplicated_documents_by_content(
    db_session: Session,
) -> dict[str, list[SourceDocument]]:
    """Return ids of documents that have the same content as other documents."""
    duplicate_content_subquery = (
        select(SourceDocument.content)
        .where(SourceDocument.content != None)
        .where(SourceDocument.content != "")
        .group_by(SourceDocument.content)
        .having(func.count(SourceDocument.content) > 1)
    )
    query = (
        select(SourceDocument)
        .where(SourceDocument.content != None)
        .where(SourceDocument.content != "")
        .where(SourceDocument.content.in_(duplicate_content_subquery))
    )
    docs = db_session.execute(query).scalars().all()
    duplicates = defaultdict(list)
    for doc in docs:
        content_fingerprint = doc.source + str(hash(doc.content.strip().lower()))
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

    logger.info(
        f"Found {len(documents_to_delete)} duplicate documents to delete: {[d.id for d in documents_to_delete]}"
    )

    for doc in documents_to_delete:
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
def drop_duplicate_documents():
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        logger.info("Getting duplicate documents by metadata")
        duplicate_documents_by_metadata = get_duplicated_documents_by_metadata(
            db_session
        )
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
    type=click.Path(exists=False, dir_okay=False),
    required=True,
    help="Sqlite3 file to read the database from",
)
def vacuum(db_file: str):
    with db_session_scope(f"sqlite:///{db_file}", preview=PREVIEW) as db_session:
        db_session.execute(text("VACUUM"))


@cli.command()
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(exists=False, dir_okay=False),
    required=True,
    help="Sqlite3 file to read the database from",
)
def clear_document_content(db_file: str):
    db_session = get_db_session(f"sqlite:///{db_file}")
    db_session.query(SourceDocument).update({"content": ""})
    db_session.commit()
    db_session.close()


@cli.command()
def drop_duplicate_chunks():
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        duplicate_chunks_by_content = get_duplicated_chunks_by_content(db_session)
        if duplicate_chunks_by_content:
            logger.info("Dropping duplicate chunks")
            drop_duplicated_chunks(db_session, duplicate_chunks_by_content)
        else:
            logger.info("No duplicate chunks found")


@cli.command()
@click.option(
    "--min-length",
    default=5,
    type=int,
    help="Minimum length of chunk content to keep",
)
@click.option(
    "--max-length",
    default=4000,
    type=int,
    help="Maximum length of chunk content to keep",
)
def cull_chunks(min_length: int, max_length: int):
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        # Select chunks that have very short content
        query = select(Chunk).where(
            or_(
                func.length(Chunk.content) < min_length,
                func.length(Chunk.content) > max_length,
            )
        )
        chunks = db_session.execute(query).scalars().all()
        logger.info(f"Found {len(chunks)} chunks to cull: {[c.id for c in chunks]}")
        if not PREVIEW:
            for chunk in chunks:
                db_session.delete(chunk)
        else:
            raise Exception("Preview mode, not deleting anything")


@cli.command()
def drop_duplicate_schemas():
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        duplicate_schemas_by_content = get_duplicated_schemas_by_content(db_session)
        if duplicate_schemas_by_content:
            drop_duplicated_schemas(db_session, duplicate_schemas_by_content)
        else:
            logger.info("No duplicate schemas found")

        if PREVIEW:
            raise Exception("Preview mode, not deleting anything")


@cli.command(
    name="drop-dangling-schemas",
)
@click.option(
    "--limit",
    default=10000,
    type=int,
    help="Maximum number of dangling schemas to drop",
)
def drop_dangling_schemas_command(limit: int):
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        logger.info("Querying dangling schemas")
        dangling_schema_ids = get_dangling_schema_ids(db_session, limit=limit)
        if not dangling_schema_ids:
            logger.info("No dangling schemas found")
            return
        logger.info(f"Dropping {len(dangling_schema_ids)} dangling schemas")
        db_session.execute(
            delete(JsonSchema).where(JsonSchema.id.in_(dangling_schema_ids))
        )


if __name__ == "__main__":
    load_dotenv(override=True)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    logger.info("Starting script")
    cli()
