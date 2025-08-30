from __future__ import annotations
import asyncio
import os
import traceback
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import TextIO
from any2json.benchmarks.models.vllm_server_mixin import VLLMServerMixin
from any2json.training.utils import format_example
from any2json.utils import logger
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
    temperature: float = 0.0
    guided_json: bool = False

    def get_state(self) -> dict:
        return vars(self)

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
            i: int, sample: dict
        ) -> tuple[str, dict] | tuple[Exception, str]:
            prompt = format_example(sample["input_data"], sample["schema"])
            payload = {
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if self.guided_json and isinstance(sample["schema"], dict):
                payload["guided_json"] = sample["schema"]

            result = {"id": i}
            try:
                async with semaphore:
                    completion, meta = await self.request_completion(payload)

                result["completion"] = completion
                result["meta"] = meta

                answer = completion["choices"][0]["text"]
                result["answer"] = answer

            except Exception as e:
                logger.error(e, exc_info=True)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback_str = "".join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                )
                result["error"] = str(e.__class__.__name__)
                result["traceback"] = traceback_str
            return result

        tasks = [task(i, sample) for i, sample in enumerate(samples)]
        results = await tqdm_asyncio.gather(*tasks, desc="Executing requests")

        errors = [result for result in results if result.get("error")]

        logger.info(
            f"Obtained {len(results)} total results: {len(results) - len(errors)} successful results, {len(errors)} errors"
        )
        return results
