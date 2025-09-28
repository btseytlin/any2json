from __future__ import annotations
import asyncio
import os
import traceback
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import TextIO
from any2json.benchmarks.models.vllm_server_mixin import VLLMServerMixin
from any2json.training.utils import format_example
from any2json.utils import json_dumps_minified, logger
import subprocess
import sys
import time
import httpx
from urllib.parse import urlparse
from tqdm.asyncio import tqdm as tqdm_asyncio


@dataclass
class VLLMServerModel(VLLMServerMixin):
    model_name: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.1
    guided_json: bool = False
    _restart_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def get_state(self) -> dict:
        state = vars(self)
        state["vllm_server_command"] = self.build_server_command()
        return state

    async def restart_server_if_needed(self) -> None:
        async with self._restart_lock:
            if not self.is_server_alive():
                logger.info("Server is not alive, restarting...")
                self.stop_server()
                self.ensure_server_started()
                logger.info("Server restarted successfully")

    async def async_get_predictions(
        self,
        samples: list[dict],
    ) -> list[dict]:
        results: list[dict] = []

        logger.info(
            f"Executing {len(samples)} requests concurrently with {self.max_concurrent_requests=}"
        )

        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def task(
            i: int, sample: dict[str, str]
        ) -> tuple[str, dict] | tuple[Exception, str]:
            prompt = format_example(sample["input_data"], sample["schema"])
            payload = {
                "prompt": prompt,
                "max_tokens": self.max_tokens,
            }
            if self.guided_json and isinstance(sample["schema"], dict):
                payload["guided_json"] = sample["schema"]

            result = {"id": i}
            max_retries = 2

            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        async with self._restart_lock:
                            pass
                        completion, meta = await self.request_completion(payload)

                    result["completion"] = completion
                    result["meta"] = meta

                    answer = completion["choices"][0]["text"]
                    result["answer"] = answer
                    break
                except Exception as e:
                    if (
                        isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code == 500
                    ):
                        logger.warning(
                            f"Got 500 error on attempt {attempt + 1}, restarting server..."
                        )
                        await self.restart_server_if_needed()
                        if attempt < max_retries - 1:
                            continue
                    logger.error(e, exc_info=True)
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback_str = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    )
                    result["error"] = {
                        "class": str(e.__class__.__name__),
                        "message": str(e),
                        "traceback": traceback_str,
                    }
                    break
            return result

        tasks = [task(i, sample) for i, sample in enumerate(samples)]
        results = await tqdm_asyncio.gather(*tasks, desc="Executing requests")

        errors = [result for result in results if result.get("error")]

        logger.info(
            f"Obtained {len(results)} total results: {len(results) - len(errors)} successful results, {len(errors)} errors"
        )
        return results
