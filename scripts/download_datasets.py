import logging
import os
import time
import click
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from any2json.utils import logger, configure_loggers


def download_huggingface_datasets(
    output_dir: str,
    max_records: int | None = None,
    hf_token: str | None = None,
    overwrite: bool = False,
):
    datasets = {
        "ChristianAzinn/json-training": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "interstellarninja/json-mode-reasoning": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "interstellarninja/json-mode-verifiable": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "interstellarninja/json-mode-agentic-reasoning": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "interstellarninja/json-schema-store-reasoning": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "mdhasnainali/job-html-to-json": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "shubh303/Invoice-to-Json": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "GokulRajaR/invoice-ocr-json": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "dataunitylab/json-schema": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "dataunitylab/json-schema-keywords": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "dataunitylab/json-schema-descriptions": {
            "args": (),
            "kwargs": {"split": "train"},
        },
        "wikimedia/structured-wikipedia": {
            "args": ("20240916.en",),
            "kwargs": {"split": "train"},
        },
    }

    for dataset_id, dataset_info in datasets.items():
        args, kwargs = dataset_info["args"], dataset_info["kwargs"]
        dataset_org, dataset_name = dataset_id.split("/")
        output_path = f"{output_dir}/{dataset_org}_{dataset_name}"
        logger.info(f"Downloading dataset {dataset_org}/{dataset_id} to {output_path}")
        if os.path.exists(output_path) and not overwrite:
            logger.info("Already exists, skipping")
            continue

        try:
            if max_records:
                dataset = load_dataset(
                    dataset_id,
                    *args,
                    **kwargs,
                    token=hf_token,
                    streaming=True,
                )
                limited_dataset = dataset.take(max_records)

                dataset_to_save = Dataset.from_generator(lambda: limited_dataset)
            else:
                dataset_to_save = load_dataset(
                    dataset_id, *args, **kwargs, token=hf_token
                )

            logger.info(
                f"Saving dataset {dataset_id} to {output_dir}/{dataset_org}_{dataset_name}"
            )
            dataset_to_save.save_to_disk(f"{output_dir}/{dataset_org}_{dataset_name}")
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_id}: {e}")
        time.sleep(10)


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
@click.option(
    "--overwrite",
    default=False,
    type=bool,
    help="Overwrite existing datasets",
)
def run(output_dir, max_records, overwrite):
    logger.info(f"Downloading datasets to {output_dir}")
    hf_token = os.getenv("HF_TOKEN")
    download_huggingface_datasets(output_dir, max_records, hf_token, overwrite)


if __name__ == "__main__":
    load_dotenv(override=True)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "INFO"),
    )
    run()
