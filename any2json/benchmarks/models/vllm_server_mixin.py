from dataclasses import field, dataclass
import os
import subprocess
import sys
import time
from typing import TextIO
from urllib.parse import urlparse
import asyncio

import httpx
import torch

from any2json.utils import logger


@dataclass
class VLLMServerMixin:
    base_url: str = "http://localhost:8000/v1"
    vllm_serve_args: list[str] = field(default_factory=lambda: ["--dtype", "auto"])
    server_process: subprocess.Popen | None = field(default=None, init=False)
    server_startup_timeout: float = 360.0
    server_log_path: str | None = "vllm_server.log"
    server_logs: str | None = None
    server_log_handle: TextIO | None = field(default=None, init=False)
    request_timeout: float = 30.0
    max_concurrent_requests: int = 6
    http_client: httpx.AsyncClient | None = field(default=None, init=False)

    def parse_host_port(self) -> tuple[str, int]:
        u = urlparse(self.base_url)
        return u.hostname, u.port

    def get_http_client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=self.request_timeout)
        return self.http_client

    async def close_http_client(self) -> None:
        if self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None

    def health_url(self) -> str:
        return f"{self.base_url}/models"

    def is_server_alive(self) -> bool:
        try:
            r = httpx.get(self.health_url(), timeout=2.0)
            logger.info(f"Health check response: {r.status_code} {r.text}")
            return r.status_code == 200
        except Exception as e:
            return False

    def build_server_command(self) -> list[str]:
        host, port = self.parse_host_port()

        args = []
        if self.vllm_serve_args:
            args += self.vllm_serve_args

        return [
            sys.executable,
            "-u",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name,
            "--host",
            host,
            "--port",
            str(port),
            *args,
        ]

    def open_log_file(self) -> None:
        if self.server_log_path:
            if os.path.exists(self.server_log_path):
                os.remove(self.server_log_path)
            self.server_log_handle = open(self.server_log_path, "ab", buffering=0)

    def close_log_file(self) -> None:
        if self.server_log_handle:
            try:
                self.server_log_handle.flush()
            except Exception:
                pass
            try:
                self.server_log_handle.close()
            except Exception:
                pass
            self.server_log_handle = None

            with open(self.server_log_path, "r") as f:
                self.server_logs = f.read()

    def spawn_server(self, cmd: list[str]) -> None:
        try:
            self.open_log_file()
            out = self.server_log_handle or subprocess.DEVNULL
            err = self.server_log_handle or subprocess.DEVNULL
            self.server_process = subprocess.Popen(cmd, stdout=out, stderr=err)
        except Exception as e:
            raise RuntimeError(f"Failed to start vLLM server: {e}")

    def wait_for_server_ready(self) -> None:
        deadline = time.time() + self.server_startup_timeout
        while time.time() < deadline:
            if self.is_server_alive():
                return
            if self.server_process:
                process_status = self.server_process.poll()
                if process_status is not None:
                    raise RuntimeError(
                        f"vLLM server exited early with status {process_status}"
                    )
            time.sleep(1)
        raise TimeoutError("Timed out waiting for vLLM server to start")

    def ensure_server_started(self) -> bool:
        logger.info("Checking if server is alive")
        if self.is_server_alive():
            logger.warning("VLLM server is already running")
            raise RuntimeError("VLLM server is already running")
        cmd = self.build_server_command()
        logger.info(f"Starting server with command: {' '.join(cmd)}")
        self.spawn_server(cmd)
        logger.info("Waiting for server to be ready")
        self.wait_for_server_ready()
        logger.info("VLLM server is ready")
        return True

    def stop_server(self) -> None:
        logger.info("Stopping server")
        if self.server_process is None:
            return
        try:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.info("Server process timed out, killing it")
                self.server_process.kill()
        finally:
            self.server_process = None
        logger.info("Server stopped")
        self.close_log_file()

    async def request_completion(
        self,
        payload: dict,
    ) -> tuple[str, dict]:
        client = self.get_http_client()
        logger.debug(f"Request payload: {payload}")
        t0 = time.perf_counter()
        response = await client.post(
            f"{self.base_url}/completions",
            json=payload,
        )
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        logger.debug(f"Request completed in {ms:.2f}ms")

        try:
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Request failed: {response.text}")
            raise e

        result = response.json()
        return result, {"inference_ms": ms}

    async def request_chat_completions(
        self,
        payload: dict,
    ) -> tuple[dict, dict]:
        client = self.get_http_client()
        t0 = time.perf_counter()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        logger.debug(f"Chat completion request completed in {ms:.2f}ms")

        try:
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Request failed: {response.text}")
            raise e

        result = response.json()
        return result, {"inference_ms": ms}

    def get_predictions(
        self,
        samples: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        started = False
        try:
            started = self.ensure_server_started()
            return asyncio.run(self.async_get_predictions(samples))
        finally:
            if started:
                self.stop_server()
