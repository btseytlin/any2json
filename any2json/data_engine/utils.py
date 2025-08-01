import json

from typing import Any
import xml
from any2json.data_engine.helpers import deduplicate_chunks
from any2json.database.models import Chunk, SourceDocument
from any2json.enums import ContentType
from any2json.utils import logger, stringify_content


def save_chunks_to_db(
    db_session,
    document_content: str,
    document_meta: dict,
    per_document_chunks: list[list[dict]],
    meta: dict,
    format: str,
):
    content_type = ContentType[format.upper()]

    new_documents = []
    new_chunks = []
    for document_content, document_meta, document_chunks in zip(
        document_content,
        document_meta,
        per_document_chunks,
        strict=True,
    ):
        document_meta.update(meta)

        document_entity = SourceDocument(
            source="infinigram",
            content="",
            content_type=ContentType.TEXT.value,
            meta=document_meta,
        )

        for chunk in document_chunks:
            content = stringify_content(chunk, format)

            chunk_entity = Chunk(
                content=content,
                content_type=content_type.value,
                is_synthetic=False,
                parent_document=document_entity,
                meta=document_meta,
            )
            new_chunks.append(chunk_entity)
        new_documents.append(document_entity)
    logger.info(
        f"Prepared {len(new_chunks)} {format} chunks and {len(new_documents)} documents."
    )

    deduplicated_chunks = deduplicate_chunks(new_chunks)
    db_session.add_all(deduplicated_chunks)


def preview_chunks(results: list[dict]):
    for result in results:
        print(f"{result['document_meta']=}")
        print(f"{result['document']=}")
        print("Chunks:")
        for i, chunk in enumerate(result["chunks"]):
            print(f"{i}: {chunk}")
            print()
        print()
        print()
