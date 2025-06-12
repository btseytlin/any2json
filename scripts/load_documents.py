import logging
import click
import os
import json
from datasets import Dataset
from dataclasses import asdict

from any2json.containers import InputJSONChunk
from any2json.database.client import create_tables, get_db_session
from any2json.database.models import SourceDocument
from any2json.enums import ContentType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_source_documents_from_wikipedia(
    dataset: Dataset, num_samples_per_dataset: int | None = None
) -> list[SourceDocument]:
    source_documents = []
    for i, record in enumerate(dataset):
        if isinstance(record, (list, dict)):
            source_documents.append(
                SourceDocument(
                    source="wikimedia_structured-wikipedia",
                    content=json.dumps(record),
                    content_type=ContentType.JSON.value,
                    meta={
                        "source_dataset_index": i,
                    },
                )
            )

        if num_samples_per_dataset and len(source_documents) >= num_samples_per_dataset:
            break

    return source_documents


def load_documents(
    dataset_dir: str, num_samples_per_dataset: int
) -> list[SourceDocument]:
    dataset_processors = {
        "wikimedia_structured-wikipedia": generate_source_documents_from_wikipedia,
    }
    processor = dataset_processors[os.path.basename(dataset_dir)]

    samples = []
    dataset = Dataset.load_from_disk(dataset_dir)
    samples.extend(processor(dataset, num_samples_per_dataset))
    return samples


@click.command()
@click.option(
    "--input-dir",
    default="data/raw",
    type=click.Path(exists=True),
    required=True,
    help="Directory to load the documents from",
)
@click.option(
    "--db-file",
    default="data/database.db",
    type=click.Path(),
    required=True,
    help="Sqlite3 file to save the database to",
)
@click.option(
    "--num-samples-per-dataset",
    default=None,
    type=int,
    help="Number of documents to load from dataset",
)
def run(input_dir: str, db_file: str, num_samples_per_dataset: int):
    db_session = get_db_session(f"sqlite:///{db_file}")

    try:
        create_tables(db_session)

        logger.info(f"Loading documents from {input_dir}")
        documents = load_documents(input_dir, num_samples_per_dataset)
        logger.info(f"Loaded {len(documents)} documents")

        db_session.add_all(documents)
        db_session.commit()

    except Exception as e:
        db_session.rollback()
        raise e
    finally:
        db_session.close()


if __name__ == "__main__":
    run()
