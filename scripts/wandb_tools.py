import os
import click
from dotenv import load_dotenv
import wandb

from any2json.utils import configure_loggers

WANDB_RUN = None


@click.group()
def cli():
    global WANDB_RUN
    WANDB_RUN = wandb.init()


@cli.command()
@click.option("--model-id", default="btseytlin/model-registry/any2json_smollm2175:v1")
@click.option(
    "--output-root", default="models", type=click.Path(dir_okay=True, file_okay=False)
)
def download_model(model_id: str, output_root: str):
    artifact = WANDB_RUN.use_artifact(model_id, type="model")
    output_dir = os.path.join(output_root, model_id.split("/")[-1])
    artifact_dir = artifact.download(output_dir)
    print(f"Model downloaded to {artifact_dir}")


if __name__ == "__main__":
    load_dotenv(override=False)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    cli()
