import os
import base64
import shlex
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import click
import runpod
from dotenv import load_dotenv, dotenv_values, find_dotenv
from any2json.utils import logger

DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"  # boristseitlin/any2json:a40
DEFAULT_GPU = "A40"


@dataclass
class PodConfig:
    name: str
    template_id: str
    gpu_type_id: str
    cloud_type: str | None
    data_center_id: str | None
    container_disk_gb: int | None
    volume_gb: int | None
    volume_mount_path: str | None
    start_ssh: bool
    ports: tuple[str, ...]
    env: dict[str, str]
    network_volume_id: str | None
    docker_args: str | None


def parse_kv_pairs(pairs: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in pairs:
        if "=" in item:
            k, v = item.split("=", 1)
            result[k] = v
    return result


def build_pod_kwargs(pcfg: PodConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "name": pcfg.name,
        "start_ssh": pcfg.start_ssh,
        "ports": ",".join(pcfg.ports) if pcfg.ports else None,
        "env": pcfg.env,
    }
    # Only add template_id if provided
    if pcfg.template_id:
        kwargs["template_id"] = pcfg.template_id
    # Only add gpu_type_id if no template is used
    if pcfg.gpu_type_id and not pcfg.template_id:
        kwargs["gpu_type_id"] = pcfg.gpu_type_id
    if pcfg.container_disk_gb is not None:
        kwargs["container_disk_in_gb"] = pcfg.container_disk_gb
    if pcfg.cloud_type:
        kwargs["cloud_type"] = pcfg.cloud_type
    if pcfg.data_center_id:
        kwargs["data_center_id"] = pcfg.data_center_id
    if pcfg.volume_gb is not None:
        kwargs["volume_in_gb"] = pcfg.volume_gb
    if pcfg.volume_mount_path:
        kwargs["volume_mount_path"] = pcfg.volume_mount_path
    if pcfg.network_volume_id:
        kwargs["network_volume_id"] = pcfg.network_volume_id
    if pcfg.docker_args:
        kwargs["docker_args"] = pcfg.docker_args
    return kwargs


@click.command()
@click.option("--name", default="any2json-train", type=str)
@click.option("--template-id", required=False, type=str)
@click.option("--gpu-type", "gpu_type", default=DEFAULT_GPU, type=str)
@click.option("--image", default=DEFAULT_IMAGE, type=str, help="Docker image to use")
@click.option("--cloud-type", default=None, type=str)
@click.option("--data-center-id", default=None, type=str)
@click.option("--container-disk-gb", default=40, type=int)
@click.option("--volume-gb", default=None, type=int)
@click.option("--volume-mount-path", default="/workspace", type=str)
@click.option("--start-ssh", is_flag=True)
@click.option("--port", "ports", multiple=True, type=str)
@click.option("--env", "env_pairs", multiple=True, type=str)
@click.option("--network-volume-id", "network_volume_id", default=None, type=str)
@click.option("--docker-args", default=None, type=str)
@click.option("--command", default=None, type=str, help="Command to run on the pod")
@click.option("--script", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--auto-terminate", is_flag=True)
def submit(
    name: str,
    template_id: str,
    gpu_type: str | None,
    cloud_type: str | None,
    data_center_id: str | None,
    container_disk_gb: int | None,
    volume_gb: int | None,
    volume_mount_path: str | None,
    start_ssh: bool,
    ports: tuple[str, ...],
    env_pairs: tuple[str, ...],
    network_volume_id: str | None,
    docker_args: str | None,
    command: str | None,
    image: str,
    script: str | None,
    auto_terminate: bool,
):
    path = find_dotenv(usecwd=True)
    load_dotenv(path)
    file_env = dotenv_values(path) if path else {}
    if not network_volume_id:
        network_volume_id = os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    cli_env = parse_kv_pairs(env_pairs)
    env = {**file_env, **cli_env}

    def select_gpu_id(preferred: str | None) -> str:
        gpus = runpod.get_gpus()
        if preferred:
            cand = preferred.replace("_", " ").strip()
            for g in gpus:
                if g["id"] == preferred or g["id"] == cand:
                    return g["id"]
            for g in gpus:
                if (
                    cand.lower() in g["id"].lower()
                    or cand.lower() in g.get("displayName", "").lower()
                ):
                    return g["id"]
        for key in ["A100", "H100", "MI300", "L40", "A40", "A30"]:
            for g in gpus:
                if key in g["id"]:
                    return g["id"]
        return gpus[0]["id"]

    gpu_type_id = select_gpu_id(gpu_type) if not template_id else None

    pcfg = PodConfig(
        name=name,
        template_id=template_id,
        gpu_type_id=gpu_type_id,
        cloud_type=cloud_type,
        data_center_id=data_center_id,
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
        volume_mount_path=volume_mount_path,
        start_ssh=start_ssh,
        ports=ports,
        env=env,
        network_volume_id=network_volume_id,
        docker_args=docker_args,
    )
    kwargs = build_pod_kwargs(pcfg)

    docker_args = kwargs.get("docker_args", "")
    cmd = ""

    if script:
        script_text = Path(script).read_text()
        encoded = base64.b64encode(script_text.encode()).decode()
        cmd = (
            f'echo \\"{encoded}\\" | base64 -d > /workspace/run.sh && '
            f"chmod +x /workspace/run.sh && /workspace/run.sh"
        ).strip()
    elif command:
        cmd = command

    if auto_terminate and cmd:
        cmd = f"{cmd} && sleep 10m && runpodctl stop pod $RUNPOD_POD_ID"

    if cmd:
        cmd = f"bash -lc '{cmd}'"

    docker_args = f"{docker_args} {cmd}".strip()

    kwargs["docker_args"] = docker_args
    logger.info(f"Creating pod with kwargs: {kwargs}")
    pod = runpod.create_pod(
        image_name=image,
        **kwargs,
    )
    click.echo(pod.get("id") or pod)


if __name__ == "__main__":
    submit()
