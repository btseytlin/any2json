import logging
import click
import os
import json
from datasets import Dataset
from dataclasses import asdict

from any2json.containers import InputJSONChunk
from any2json.data_engine.utils import generate_chunks_from_json_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_input_json_chunks(
    input_dir: str, num_samples_per_dataset: int
) -> list[InputJSONChunk]:
    dataset_processors = {
        "wikimedia_structured-wikipedia": generate_chunks_from_json_dataset,
    }
    samples = []
    for hf_dataset in os.scandir(input_dir):
        if not hf_dataset.is_dir() or hf_dataset.name.startswith("."):
            continue

        dataset = Dataset.load_from_disk(hf_dataset.path)
        processor = dataset_processors[hf_dataset.name]
        samples.extend(processor(dataset, hf_dataset.name, num_samples_per_dataset))
    return samples


@click.command()
@click.option(
    "--input-dir",
    default="data/raw",
    type=click.Path(exists=True),
    required=True,
    help="Directory to save the inputs",
)
@click.option(
    "--output-dir",
    default="data/intermediate",
    type=click.Path(),
    required=True,
    help="Directory to save the samples",
)
@click.option(
    "--num-samples-per-dataset",
    default=100,
    type=int,
    help="Number of samples to generate from each dataset",
)
def run(input_dir: str, output_dir: str, num_samples_per_dataset: int):
    logger.info(f"Generating input chunks from {input_dir}")
    chunks = generate_input_json_chunks(input_dir, num_samples_per_dataset)
    for i, chunk in enumerate(chunks):
        chunk.id = i
    logger.info(f"Generated {len(chunks)} input chunks")

    os.makedirs(output_dir, exist_ok=True)

    chunk_dicts = [asdict(chunk) for chunk in chunks]
    with open(f"{output_dir}/input_json_chunks.json", "w") as f:
        json.dump(chunk_dicts, f, indent=2)


if __name__ == "__main__":
    run()
