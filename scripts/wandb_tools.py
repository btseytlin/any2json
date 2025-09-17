import logging
import os
import click
from dotenv import load_dotenv
import wandb

from any2json.utils import configure_loggers, logger

WANDB_RUN = None


@click.group()
@click.option("--run-id", help="Existing wandb run ID to resume", envvar="WANDB_RUN_ID")
@click.option("--quiet", is_flag=True, help="Quiet mode")
def cli(run_id: str, quiet: bool):
    if quiet:
        wandb_logger = logging.getLogger("wandb")
        wandb_logger.setLevel(logging.ERROR)
        os.environ["WANDB_SILENT"] = "true"

    global WANDB_RUN
    if run_id:
        WANDB_RUN = wandb.init(id=run_id, resume="allow")
        logger.debug(f"Resumed wandb run: {run_id}")
    else:
        WANDB_RUN = wandb.init()
        logger.debug(f"Started new wandb run: {WANDB_RUN.id}")


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


@cli.command()
@click.argument("artifact-id", type=str)
@click.option(
    "--output-root",
    default="artifacts",
    type=click.Path(dir_okay=True, file_okay=False),
)
def download_artifact(artifact_id: str, output_root: str):
    artifact = WANDB_RUN.use_artifact(artifact_id)
    output_dir = os.path.join(output_root, artifact_id.split("/")[-1])
    artifact_dir = artifact.download(output_dir)
    print(f"Artifact downloaded to {artifact_dir}")


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option("--name", required=True, help="Name for the artifact")
@click.option("--type", "artifact_type", default="dataset", help="Type of the artifact")
@click.option("--description", help="Description for the artifact", default="")
@click.option("--incremental", is_flag=True, help="Enable incremental updates")
def upload_directory(
    directory: str, name: str, artifact_type: str, description: str, incremental: bool
):
    try:
        if incremental:
            existing_artifact = WANDB_RUN.use_artifact(
                f"{name}:latest", type=artifact_type
            )
            artifact = wandb.Artifact(
                name=name, type=artifact_type, description=description
            )
            for entry in existing_artifact.manifest.entries.values():
                if not os.path.exists(os.path.join(directory, entry.path)):
                    artifact.add_reference(existing_artifact.get_entry(entry.path).ref)
            print(
                f"Using existing artifact {name}:latest as base for incremental update"
            )
        else:
            artifact = wandb.Artifact(
                name=name, type=artifact_type, description=description
            )
    except wandb.errors.CommError:
        raise RuntimeError(f"No existing artifact {name} found")

    artifact.add_dir(directory)
    WANDB_RUN.log_artifact(artifact)
    logger.info(f"Directory {directory} uploaded as artifact {name}")


@cli.command()
def get_run_id():
    click.echo(WANDB_RUN.id)


if __name__ == "__main__":
    load_dotenv(override=False)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    cli()
