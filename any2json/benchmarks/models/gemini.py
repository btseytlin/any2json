from __future__ import annotations
import asyncio
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from tqdm.asyncio import tqdm as tqdm_asyncio
from any2json.utils import logger
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
)


def to_text(x: str | dict) -> str:
    return json.dumps(x) if isinstance(x, dict) else str(x)


def clean_answer_text(text: str) -> str:
    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def system_prompt() -> str:
    return """
Your task is given a JSONSchema and an input text, convert the input data to JSON according to the JSONSchema.

Output only the resulting JSON, nothing else.

## Example
### JSONSchema
{
    "type": ["object", "null"],
    "properties": {
        "name": {
            "type": ["string", "null"],
        },
        "age": {
            "type": ["integer", "null"],
        },
        "isStudent": {
            "type": ["boolean", "null"],
        },
        "courses": {
            "type": ["array", "null"],
            "items": {
                "type": ["string", "null"],
            },
        }
    },
}

### Input
name: Sam Smith
age: 25
isStudent: true
courses: ["Math", "Science"]

### Output
{
    "name": "Sam Smith",
    "age": 25,
    "isStudent": true,
    "courses": ["Math", "Science"]
}
"""


def make_prompt(input_text: str, schema: dict) -> str:
    return f"""
## Task
### JSONSchema
{json.dumps(schema)}

### Input
{input_text}

### Output
"""


class GeminiAgentDeps(BaseModel):
    input_text: str = Field(...)
    schema: dict = Field(...)


class GeminiAgentOut(BaseModel):
    answer: str = Field(...)


@dataclass
class GeminiModel:
    model_name: str = "gemini-2.5-flash-lite"
    enable_thinking: bool = False
    max_tokens: int = 8000
    thinking_budget: int = 1024
    include_thoughts: bool = False
    max_concurrent_tasks: int = 20
    request_max_retries: int = 3
    backoff_min_s: float = 1.0
    backoff_max_s: float = 30.0
    backoff_multiplier: float = 2.0
    request_timeout_s: float = 20
    agent: Agent = field(init=False)

    def __post_init__(self) -> None:
        model = GoogleModel(
            self.model_name,
            settings=(
                GoogleModelSettings(
                    google_thinking_config={
                        "thinking_budget": self.thinking_budget,
                        "include_thoughts": self.include_thoughts,
                    },
                    timeout=self.request_timeout_s,
                )
                if self.enable_thinking
                else None
            ),
        )
        self.agent = Agent(
            model=model,
            system_prompt=system_prompt(),
            deps_type=GeminiAgentDeps,
            output_type=GeminiAgentOut,
            retries=2,
        )

        @self.agent.system_prompt
        async def add_input(ctx: RunContext[GeminiAgentDeps]) -> str:
            return make_prompt(ctx.deps.input_text, ctx.deps.schema)

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or "gemini-2.5-flash-lite"

    def get_state(self) -> dict:
        return {
            "model_name": self.resolved_model_name,
            "class_name": str(self.__class__.__name__),
            "enable_thinking": self.enable_thinking,
            "thinking_budget": self.thinking_budget,
            "include_thoughts": self.include_thoughts,
            "max_tokens": self.max_tokens,
        }

    async def generate_async(self, input_text: str, schema: dict) -> tuple[str, dict]:
        res = await self.agent.run(
            deps=GeminiAgentDeps(input_text=input_text, schema=schema)
        )
        content = clean_answer_text(res.output.answer)
        model_used = res.all_messages()[-2].model_name
        return content, {"model_name": model_used}

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, dict]:
        return asyncio.run(self.generate_async(input_text, schema))

    async def _get_predictions_async(
        self, samples: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        results: list[dict] = []
        errors: list[dict] = []
        sem = asyncio.Semaphore(self.max_concurrent_tasks)

        async def run_one(i: int, s: dict) -> tuple[str, int, str, dict | None]:
            async with sem:
                try:
                    # async for attempt in AsyncRetrying(
                    #     retry=retry_if_exception_type(Exception),
                    #     wait=wait_exponential(
                    #         multiplier=self.backoff_multiplier,
                    #         min=self.backoff_min_s,
                    #         max=self.backoff_max_s,
                    #     ),
                    #     stop=stop_after_attempt(self.request_max_retries),
                    #     before_sleep=before_sleep_log(logger, logging.ERROR),
                    # ):
                    #     with attempt:
                    x = to_text(s["input_data"])
                    schema = json.loads(s["schema"])
                    ans, meta = await self.generate_async(x, schema)
                    return "ok", i, ans, meta
                except Exception as e:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    tb = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_tb)
                    ).strip()
                    logger.error(f"Error generating for sample {i}: {e}")
                    logger.error(tb)
                    return "err", i, str(e), {"traceback": tb}

        tasks = [run_one(i, s) for i, s in enumerate(samples)]
        for fut in tqdm_asyncio.as_completed(tasks):
            status, i, a, meta = await fut
            if status == "ok":
                results.append({"id": i, "answer": a, "meta": meta or {}})
            else:
                errors.append(
                    {
                        "id": i,
                        "error": a,
                        "traceback": (meta or {}).get("traceback", ""),
                    }
                )
        results.sort(key=lambda x: x["id"])
        errors.sort(key=lambda x: x["id"])  # type: ignore
        return results, errors

    def get_predictions(
        self, samples: list[dict], **_: Any
    ) -> tuple[list[dict], list[dict]]:
        return asyncio.run(self._get_predictions_async(samples))
