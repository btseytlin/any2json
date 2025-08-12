from __future__ import annotations
import asyncio
import json
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import tqdm.asyncio as tqdm_asyncio
from any2json.utils import logger


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
    return (
        "You are a helpful assistant that can convert structured data to JSON according to the provided JSONSchema.\n\n"
        "## Task:\n"
        "Convert the input data to JSON according to the JSONSchema.\n"
        "If a property is not present in the input data, it should be present in the output and set to null.\n"
        'Ignore the "required" field in the JSONSchema.\n'
        "Return the resulting JSON object only, without any other text."
    )


def make_prompt(input_text: str, schema: dict) -> str:
    return f"""
        ## Input data:
        {input_text}

        ## JSONSchema:
        {json.dumps(schema)}
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
    max_concurrent_tasks: int = 100
    agent: Agent = field(init=False)

    def __post_init__(self) -> None:
        model = GoogleModel(
            self.model_name,
            settings=(
                GoogleModelSettings(
                    google_thinking_config={
                        "thinking_budget": self.thinking_budget,
                        "include_thoughts": self.include_thoughts,
                    }
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
            retries=3,
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
            "max_tokens": self.max_tokens,
        }

    async def generate_async(self, input_text: str, schema: dict) -> tuple[str, dict]:
        result = await self.agent.run(
            deps=GeminiAgentDeps(input_text=input_text, schema=schema)
        )
        content = clean_answer_text(result.output.answer)
        model_used = result.all_messages()[-2].model_name
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
                    x = to_text(s["input_data"])
                    ans, meta = await self.generate_async(x, s["schema"])
                    return "ok", i, ans, meta
                except Exception as e:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    tb = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_tb)
                    ).strip()
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
