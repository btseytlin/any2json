from collections import defaultdict
import copy
import json
import os
from dotenv import load_dotenv
import click
import fastjsonschema
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
    update,
)
from sqlalchemy.orm import Session
from tqdm import tqdm

from any2json.database.client import create_tables, db_session_scope, get_db_session
from any2json.database.helpers import get_dangling_schema_ids
from any2json.database.models import Chunk, JsonSchema, SchemaConversion, SourceDocument
from any2json.enums import ContentType
from any2json.utils import configure_loggers, logger


PREVIEW = False
DB_FILE = "data/database.db"


@click.group()
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(dir_okay=False, writable=True),
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

    document_ids_to_delete = set()
    for _, documents in duplicate_documents.items():
        for document in documents[1:]:
            document_ids_to_delete.add(document.id)

    logger.info(f"Found {len(document_ids_to_delete)} duplicate documents to delete")

    db_session.execute(
        delete(SourceDocument).where(SourceDocument.id.in_(document_ids_to_delete))
    )


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

    chunk_ids_to_delete = set()
    for _, chunks in duplicate_chunks.items():
        for chunk in chunks[1:]:
            chunk_ids_to_delete.add(chunk.id)

    logger.info(f"Found {len(chunk_ids_to_delete)} duplicate chunks to delete")

    db_session.execute(delete(Chunk).where(Chunk.id.in_(chunk_ids_to_delete)))


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
    limit: int | None = None,
):
    if not duplicate_schemas:
        return

    schema_ids_to_delete = set()
    for _, schemas in duplicate_schemas.items():
        original_schema = schemas[0]
        for schema in schemas[1:]:
            for chunk in schema.chunks:
                logger.debug(
                    f"Chunk {chunk.id} schema reassigned {chunk.schema_id} -> {original_schema.id}"
                )
                chunk.schema = original_schema
                db_session.add(chunk)
            schema_ids_to_delete.add(schema.id)
            if limit and len(schema_ids_to_delete) >= limit:
                break
        if limit and len(schema_ids_to_delete) >= limit:
            break

    logger.info(f"Found {len(schema_ids_to_delete)} duplicate schemas to delete")

    db_session.execute(
        delete(JsonSchema).where(JsonSchema.id.in_(schema_ids_to_delete))
    )


@cli.command()
def init():
    db_session = get_db_session(f"sqlite:///{DB_FILE}")
    create_tables(db_session)
    db_session.commit()
    db_session.close()


@cli.command()
@click.argument("sql_file", type=click.Path(exists=True, dir_okay=False))
def execute_sql(sql_file: str):
    with open(sql_file, "r") as f:
        sql = f.read()
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        db_session.execute(text(sql))


@cli.command()
def stats():
    """Calculate:
    - Number of source documents,
      - Number of documents with no chunks
      - By source dataset
    - Number of chunks
      - By content type
      - By synthetic/real
      - Json has schema/no schema
    - Number of schemas
      - By source dataset
      - By synthetic/real
      - JsonSchema has chunks/no chunks
    - Number of schema conversions
      - By input type
    """
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        total_docs = db_session.execute(select(func.count(SourceDocument.id))).scalar()

        used_document_ids = (
            db_session.execute(
                select(Chunk.parent_document_id).group_by(Chunk.parent_document_id)
            )
            .scalars()
            .all()
        )

        docs_no_chunks = db_session.execute(
            select(func.count(SourceDocument.id)).where(
                SourceDocument.id.not_in(used_document_ids)
            )
        ).scalar()

        docs_by_source = db_session.execute(
            select(SourceDocument.source, func.count(SourceDocument.id)).group_by(
                SourceDocument.source
            )
        ).all()

        total_chunks = db_session.execute(select(func.count(Chunk.id))).scalar()

        chunks_by_type = db_session.execute(
            select(Chunk.content_type, func.count(Chunk.id)).group_by(
                Chunk.content_type
            )
        ).all()

        chunks_by_synthetic = db_session.execute(
            select(Chunk.is_synthetic, func.count(Chunk.id)).group_by(
                Chunk.is_synthetic
            )
        ).all()

        json_chunks_with_schema = db_session.execute(
            select(func.count(Chunk.id))
            .where(Chunk.content_type == ContentType.JSON.value)
            .where(Chunk.schema_id.is_not(None))
        ).scalar()

        json_chunks_without_schema = db_session.execute(
            select(func.count(Chunk.id))
            .where(Chunk.content_type == ContentType.JSON.value)
            .where(Chunk.schema_id.is_(None))
        ).scalar()

        total_schemas = db_session.execute(select(func.count(JsonSchema.id))).scalar()

        schemas_by_synthetic = db_session.execute(
            select(JsonSchema.is_synthetic, func.count(JsonSchema.id)).group_by(
                JsonSchema.is_synthetic
            )
        ).all()

        schemas_with_chunks = db_session.execute(
            select(func.count(distinct(JsonSchema.id)))
            .select_from(JsonSchema)
            .join(Chunk, JsonSchema.id == Chunk.schema_id)
        ).scalar()

        schemas_without_chunks = total_schemas - schemas_with_chunks

        total_conversions = db_session.execute(
            select(func.count(SchemaConversion.id))
        ).scalar()

        conversions_by_input_type = db_session.execute(
            select(Chunk.content_type, func.count(SchemaConversion.id))
            .select_from(SchemaConversion)
            .join(Chunk, SchemaConversion.input_chunk_id == Chunk.id)
            .group_by(Chunk.content_type)
        ).all()

        print("=== DATABASE STATISTICS ===")
        print(f"\nSOURCE DOCUMENTS:")
        print(f"  Total: {total_docs}")
        print(f"  Documents with no chunks: {docs_no_chunks}")
        print("  By source dataset:")
        for source, count in docs_by_source:
            print(f"    {source}: {count}")

        print(f"\nCHUNKS:")
        print(f"  Total: {total_chunks}")
        print("  By content type:")
        for content_type, count in chunks_by_type:
            print(f"    {content_type}: {count}")
        print("  By synthetic/real:")
        for is_synthetic, count in chunks_by_synthetic:
            status = "synthetic" if is_synthetic else "real"
            print(f"    {status}: {count}")
        print("  JSON chunks:")
        print(f"    With schema: {json_chunks_with_schema}")
        print(f"    Without schema: {json_chunks_without_schema}")

        print(f"\nSCHEMAS:")
        print(f"  Total: {total_schemas}")
        print("  By synthetic/real:")
        for is_synthetic, count in schemas_by_synthetic:
            status = "synthetic" if is_synthetic else "real"
            print(f"    {status}: {count}")
        print("  Usage:")
        print(f"    With chunks: {schemas_with_chunks}")
        print(f"    Without chunks: {schemas_without_chunks}")

        print(f"\nSCHEMA CONVERSIONS:")
        print(f"  Total: {total_conversions}")
        print("  By input type:")
        for input_type, count in conversions_by_input_type:
            print(f"    {input_type}: {count}")


@cli.command()
def vacuum():
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        db_session.execute(text("VACUUM"))


@cli.command()
def clear_document_content():
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        db_session.execute(
            update(SourceDocument)
            .where(SourceDocument.content != "")
            .values({"content": ""})
        )


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
        query = select(Chunk).where(
            or_(
                func.length(Chunk.content) <= min_length,
                func.length(Chunk.content) >= max_length,
            )
        )
        chunks = db_session.execute(query).scalars().all()
        logger.info(f"Found {len(chunks)} chunks to cull: {[c.id for c in chunks]}")

        for chunk in chunks:
            logger.info(f"Culling chunk {chunk.id}: {chunk.meta} {chunk.content}")

        for chunk in chunks:
            db_session.delete(chunk)


@cli.command()
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Maximum number of duplicate schemas to drop",
)
def drop_duplicate_schemas(limit: int):
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        duplicate_schemas_by_content = get_duplicated_schemas_by_content(db_session)
        if duplicate_schemas_by_content:
            drop_duplicated_schemas(
                db_session,
                duplicate_schemas_by_content,
                limit=limit,
            )
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


@cli.command(
    name="drop-broken-schemas",
)
def drop_broken_schemas_command():
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        logger.info("Querying schemas")
        schemas = db_session.execute(select(JsonSchema)).scalars().all()
        broken_schemas = []
        for schema in schemas:
            try:
                fastjsonschema.compile(schema.content)
            except Exception as e:
                broken_schemas.append((schema, e))
        logger.info(f"Found {len(broken_schemas)} broken schemas")
        for schema, error in broken_schemas:
            logger.info(f"Dropping broken schema {schema.id}: {error}")
            db_session.delete(schema)


@cli.command()
def drop_xml():
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        xml_chunks = (
            db_session.execute(
                select(Chunk).where(Chunk.content_type == ContentType.XML.value)
            )
            .scalars()
            .all()
        )

        xml_chunk_ids = [chunk.id for chunk in xml_chunks]
        schema_conversions = (
            db_session.execute(
                select(SchemaConversion).where(
                    SchemaConversion.input_chunk_id.in_(xml_chunk_ids)
                )
            )
            .scalars()
            .all()
        )

        chunk_ids_to_delete = set()
        schema_ids_to_delete = set()
        schema_conversion_ids_to_delete = set()

        for schema_conversion in schema_conversions:
            chunk_ids_to_delete.add(schema_conversion.input_chunk_id)
            chunk_ids_to_delete.add(schema_conversion.output_chunk_id)
            schema_ids_to_delete.add(schema_conversion.schema_id)
            schema_conversion_ids_to_delete.add(schema_conversion.id)

        logger.info(f"Dropping {len(chunk_ids_to_delete)} chunks")
        logger.info(f"Dropping {len(schema_ids_to_delete)} schemas")
        logger.info(
            f"Dropping {len(schema_conversion_ids_to_delete)} schema conversions"
        )
        db_session.execute(delete(Chunk).where(Chunk.id.in_(chunk_ids_to_delete)))
        db_session.execute(
            delete(JsonSchema).where(JsonSchema.id.in_(schema_ids_to_delete))
        )
        db_session.execute(
            delete(SchemaConversion).where(
                SchemaConversion.id.in_(schema_conversion_ids_to_delete)
            )
        )


@cli.command()
def clear_broken_schema_mappings():
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        chunks = (
            db_session.execute(
                select(Chunk)
                .where(Chunk.schema_id.is_not(None))
                .where(Chunk.content_type == ContentType.JSON.value)
            )
            .scalars()
            .all()
        )
        total_errors = 0
        for chunk in tqdm(chunks):
            schema = chunk.schema
            if schema is None:
                logger.warning(
                    f"Chunk {chunk.id} has no schema even though {chunk.schema_id=}"
                )
                chunk.schema_id = None
                db_session.add(chunk)
                continue

            try:
                compiled_schema = fastjsonschema.compile(
                    schema.content,
                )
                compiled_schema(json.loads(chunk.content))
            except Exception as e:
                logger.warning(
                    f"Chunk {chunk.id} has schema {schema.id} but validation failed: {e}"
                )
                logger.error(e, exc_info=True)
                total_errors += 1
                chunk.schema_id = None
                db_session.add(chunk)

        logger.info(f"Total errors: {total_errors}")


@cli.command()
@click.option(
    "--delete-invalid",
    is_flag=True,
    help="Delete invalid schema conversions",
)
def validate_schema_conversions(delete_invalid: bool):
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        valid_schema_conversions = 0
        invalid_schema_conversions = 0
        for schema_conversion in tqdm(
            db_session.execute(select(SchemaConversion)).scalars()
        ):
            if (
                schema_conversion.schema is None
                or schema_conversion.output_chunk is None
                or schema_conversion.input_chunk is None
            ):
                invalid_schema_conversions += 1
                if delete_invalid:
                    db_session.delete(schema_conversion)
            else:
                valid_schema_conversions += 1
        logger.info(f"Valid schema conversions: {valid_schema_conversions}")
        logger.info(f"Invalid schema conversions: {invalid_schema_conversions}")


@cli.command()
def fix_schema_conversions():
    with db_session_scope(f"sqlite:///{DB_FILE}", preview=PREVIEW) as db_session:
        schema_conversions = (
            db_session.execute(
                select(SchemaConversion, Chunk)
                .where(SchemaConversion.schema_id != Chunk.schema_id)
                .join(Chunk, SchemaConversion.output_chunk_id == Chunk.id)
            )
            .scalars()
            .all()
        )
        for schema_conversion in tqdm(schema_conversions):
            schema_conversion.schema = schema_conversion.output_chunk.schema
            db_session.add(schema_conversion)
            logger.info(
                f"Updated schema conversion {schema_conversion.id} with to have schema id {schema_conversion.output_chunk.schema_id}"
            )
        logger.info(f"Total schema conversions updated: {len(schema_conversions)}")


if __name__ == "__main__":
    load_dotenv(override=False)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    logger.info("Starting script")
    cli()
