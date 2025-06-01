import logging
import click
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def download_huggingface_datasets(output_dir: str, max_records: int | None = None):
    datasets = {
        "wikimedia/structured-wikipedia": {
            "args": ("20240916.en",),
            "kwargs": {"split": "train"},
        }
    }

    for dataset_id, dataset_info in datasets.items():
        args, kwargs = dataset_info["args"], dataset_info["kwargs"]
        dataset_org, dataset_name = dataset_id.split("/")
        if max_records:
            dataset = load_dataset(dataset_id, *args, **kwargs, streaming=True)
            limited_dataset = dataset.take(max_records)

            dataset_to_save = Dataset.from_generator(lambda: limited_dataset)
        else:
            dataset_to_save = load_dataset(dataset_id, *args, num_proc=-1)

        logger.info(
            f"Saving dataset {dataset_id} to {output_dir}/{dataset_org}_{dataset_name}"
        )
        dataset_to_save.save_to_disk(f"{output_dir}/{dataset_org}_{dataset_name}")


@click.command()
@click.option(
    "--output-dir",
    default="data/raw",
    type=click.Path(exists=True),
    required=True,
    help="Directory to save the inputs",
)
@click.option(
    "--max-records",
    default=None,
    type=int,
    help="Maximum number of records to download from each dataset (for partial downloads)",
)
def run(output_dir, max_records):
    logger.info(f"Downloading datasets to {output_dir}")
    download_huggingface_datasets(output_dir, max_records)


if __name__ == "__main__":
    run()
