from __future__ import annotations
import asyncio
import os
import traceback
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import TextIO
from any2json.training.utils import format_example
from any2json.utils import logger
import subprocess
import sys
import time
import httpx
from urllib.parse import urlparse
from tqdm.asyncio import tqdm as tqdm_asyncio


@dataclass
class VLLMServerModel:
    model_name: str | None = None
    max_tokens: int = 8096
    temperature: float = 0.0
    guided_json: bool = False
    base_url: str = "http://localhost:8000/v1"
    vllm_serve_args: list[str] = field(default_factory=list)
    batch_size: int = 16
    server_process: subprocess.Popen | None = field(default=None, init=False)
    server_startup_timeout: float = 180.0
    server_log_path: str | None = "vllm_server.log"
    server_log_handle: TextIO | None = field(default=None, init=False)
    request_timeout: float = 180.0

    def get_state(self) -> dict:
        return vars(self)

    def __post_init__(self) -> None:
        name = self.model_name

    def parse_host_port(self) -> tuple[str, int]:
        u = urlparse(self.base_url)
        return u.hostname, u.port

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
        prompt: str,
        json_schema: dict | None = None,
    ) -> tuple[str, dict]:
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }

        if json_schema:
            payload["guided_json"] = json_schema

        with httpx.Client(timeout=self.request_timeout) as client:
            t0 = time.perf_counter()
            response = client.post(
                f"{self.base_url}/completions",
                json=payload,
            )
            t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        logger.info(f"Request completed in {ms:.2f}ms")

        response.raise_for_status()

        result = response.json()
        completion = result["choices"][0]["text"]
        return completion, {"inference_ms": ms}

    def get_predictions(
        self, samples: list[dict], max_concurrent_requests: int = 8
    ) -> tuple[list[dict], list[dict]]:
        started = False
        try:
            started = self.ensure_server_started()
            return asyncio.run(
                self.async_get_predictions(samples, max_concurrent_requests)
            )
        finally:
            if started:
                self.stop_server()

    async def async_get_predictions(
        self, samples: list[dict], max_concurrent_requests: int = 8
    ) -> tuple[list[dict], list[dict]]:
        results: list[dict] = []
        errors: list[dict] = []

        logger.info(
            f"Executing {len(samples)} requests concurrently with {max_concurrent_requests=}"
        )

        semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def task(sample: dict) -> tuple[str, dict] | tuple[Exception, str]:
            async with semaphore:
                try:
                    prompt = format_example(sample["input_data"], sample["schema"])
                    answer, meta = await self.request_completion(
                        prompt, sample["schema"]
                    )
                    return answer, meta
                except Exception as e:
                    logger.error(e)
                    return e, traceback.format_exc()

        tasks = [task(sample) for sample in samples]
        outputs = await tqdm_asyncio.gather(*tasks, desc="Executing requests")

        results = []
        for i, result in enumerate(outputs):
            if isinstance(result[0], Exception):
                exception, traceback_str = result
                errors.append(
                    {
                        "id": i,
                        "error": str(exception),
                        "traceback": traceback_str.strip(),
                    }
                )
            else:
                answer, meta = result
                results.append({"id": i, "answer": answer, "meta": meta})

        logger.info(f"Obtained {len(results)} results and {len(errors)} errors")
        return results, errors
