from __future__ import annotations
import asyncio
import traceback
from tqdm import tqdm
import json
import torch
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from any2json.benchmarks.models.vllm_server_mixin import VLLMServerMixin
from any2json.utils import logger
from tqdm.asyncio import tqdm as tqdm_asyncio
import sys


def to_text(x: str | dict) -> str:
    return json.dumps(x) if isinstance(x, dict) else str(x)


def get_sampling_params(enable_thinking: bool, max_tokens: int) -> dict:
    if enable_thinking:
        temperature = 0.6
        top_p = 0.95
        top_k = 20
        min_p = 0.0
    else:
        temperature = 0.7
        top_p = 0.8
        top_k = 20
        min_p = 0.0

    return dict(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
    )


def system_prompt() -> str:
    return """
    You are a helpful assistant that can convert structured data to JSON according to the provided JSONSchema.

    ## Task:
    Convert the input data to JSON according to the JSONSchema.
    If a property is not present in the input data, it should be present in the output and set to null.
    Ignore the "required" field in the JSONSchema.
    Return the resulting JSON object only, without any other text.
    """


def make_prompt(input_text: str, schema: dict) -> str:
    return f"""
        ## Input data:
        {input_text}

        ## JSONSchema:
        {schema}
        """


def parse_think(text: str) -> tuple[str, str]:
    if "<think>" in text and "</think>" in text:
        start = text.find("<think>") + 7
        end = text.rfind("</think>")
        return text[end + 8 :].strip(), text[start:end].strip()
    return text.strip(), ""


def clean_answer_text(text: str) -> str:
    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    s = s.replace("```", "").strip()
    tokens = ["<|im_end|>", "<|assistant|>", "<|user|>", "<|system|>"]
    for t in tokens:
        s = s.replace(t, "")
    return s.strip()


def normalize_output_text(text: str) -> str:
    return clean_answer_text(text)


def messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt().strip()},
        {"role": "user", "content": prompt.strip()},
    ]


@dataclass
class QwenVLLMServer(VLLMServerMixin):
    model_name: str = "Qwen/Qwen3-0.6B"
    enable_thinking: bool = False
    guided_json: bool = False
    tokenizer: AutoTokenizer = field(init=False)

    max_tokens: int = 4096
    temperature: float = 0.0

    def get_state(self) -> dict:
        return vars(self)

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.enable_thinking:
            self.vllm_serve_args += ["--reasoning-parser", "deepseek_r1"]

    async def async_get_predictions(self, samples: list[dict]) -> list[dict]:
        logger.info(
            f"Executing {len(samples)} requests concurrently with {self.max_concurrent_requests=}"
        )

        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def task(i: int, sample: dict) -> dict:
            x = to_text(sample["input_data"])
            prompt = make_prompt(x, sample["schema"])

            params = get_sampling_params(self.enable_thinking, self.max_tokens)
            payload = {
                "model": self.model_name,
                "messages": messages(prompt),
                **params,
            }

            if self.guided_json and isinstance(sample["schema"], dict):
                payload["guided_json"] = sample["schema"]

            result = {"id": i}

            try:
                async with semaphore:
                    completion_messages, meta = await self.request_chat_completions(
                        payload
                    )
                result["completion"] = completion_messages
                result["meta"] = meta

                message = completion_messages["choices"][0]["message"]
                answer = message.get("content")
                reasoning = message.get("reasoning_content")

                if not reasoning:
                    answer, reasoning = parse_think(answer)

                content = normalize_output_text(answer)
                result["meta"]["thinking_content"] = reasoning
                result["answer"] = content
            except Exception as e:
                logger.error(e, exc_info=True)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback_str = "".join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                )
                result["error"] = str(e)
                result["traceback"] = traceback_str
            return result

        tasks = [task(i, sample) for i, sample in enumerate(samples)]
        results = await tqdm_asyncio.gather(*tasks, desc="Executing requests")

        logger.info(f"Obtained {len(results)} results")
        return results
