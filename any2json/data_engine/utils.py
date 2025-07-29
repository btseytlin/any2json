import json
from any2json.data_engine.helpers import deduplicate_chunks
from any2json.database.models import Chunk, SourceDocument
from any2json.enums import ContentType
from any2json.utils import logger


def save_chunks_to_db(db_session, results: list[dict], meta: dict):
    new_documents = []
    new_json_chunks = []
    for result in results:
        document = result["document"]
        json_chunks = result["json_chunks"]
        document_meta = result["document_meta"]
        document_meta.update(meta)

        document_entity = SourceDocument(
            source="infinigram",
            content=document,
            content_type=ContentType.TEXT.value,
            meta=document_meta,
        )

        for json_chunk in json_chunks:
            json_chunk_entity = Chunk(
                content=json.dumps(json_chunk, indent=2, ensure_ascii=False),
                content_type=ContentType.JSON.value,
                is_synthetic=False,
                parent_document=document_entity,
                meta=document_meta,
            )
            new_json_chunks.append(json_chunk_entity)
        new_documents.append(document_entity)
    logger.info(
        f"Prepared {len(new_json_chunks)} json chunks and {len(new_documents)} documents."
    )

    deduplicated_chunks = deduplicate_chunks(new_json_chunks)
    db_session.add_all(deduplicated_chunks)


def preview_chunks(results: list[dict]):
    for result in results:
        print(f"{result['document_meta']=}")
        print(f"{result['document']=}")
        print()
        for json_chunk in result["json_chunks"]:
            print(f"{json.dumps(json_chunk, indent=2, ensure_ascii=False)}")
            print()
        print()
        print()
        print()
