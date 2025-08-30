from __future__ import annotations
import asyncio
import traceback
from tqdm import tqdm
import json
import torch
from dataclasses import dataclass, field
from typing import Iterator, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer
from any2json.utils import logger
from any2json.benchmarks.models.vllm_custom import VLLMServerModel
from tqdm.asyncio import tqdm as tqdm_asyncio
import sys
import time


def to_text(x: str | dict) -> str:
    return json.dumps(x) if isinstance(x, dict) else str(x)


def build_chat_text(
    tokenizer: AutoTokenizer, enable_thinking: bool, input_text: str, schema: dict
) -> str:
    return tokenizer.apply_chat_template(
        messages(make_prompt(input_text, schema)),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_chat_texts(
    tokenizer: AutoTokenizer, enable_thinking: bool, batch_samples: list[dict]
) -> list[str]:
    return [
        build_chat_text(
            tokenizer, enable_thinking, to_text(s["input_data"]), s["schema"]
        )
        for s in batch_samples
    ]


def hf_tokenize_to_device(
    tokenizer: AutoTokenizer, texts: list[str], device_obj: torch.device
) -> object:
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return enc.to(device_obj)


def hf_decode_batch(
    tokenizer: AutoTokenizer, outputs: object, input_lengths: list[int]
) -> list[str]:
    decoded: list[str] = []
    for i in range(len(input_lengths)):
        t = tokenizer.decode(
            outputs[i][int(input_lengths[i]) :], skip_special_tokens=True
        ).strip()
        decoded.append(t)
    return decoded


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


def parallel_map(
    fn, items: list, max_workers: int
) -> tuple[list[tuple[int, object]], list[tuple[int, tuple[Exception, str]]]]:
    results: list[tuple[int, object]] = []
    errors: list[tuple[int, Exception]] = []
    pbar = tqdm(total=len(items), desc="Executing tasks")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, i, item): i for i, item in items}
        for f in as_completed(futs):
            try:
                i = futs[f]
                try:
                    results.append((i, f.result()))
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback_str = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    )
                    logger.error(f"Error during inference: {e}")
                    errors.append((i, (e, traceback_str)))
                pbar.update(1)

            except KeyboardInterrupt:
                logger.error("Keyboard interrupt")
                for f in futs.values():
                    f.cancel()
                break
    results.sort(key=lambda x: x[0])
    errors.sort(key=lambda x: x[0])
    return results, errors


device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else device


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
class QwenVLLMServer(VLLMServerModel):
    model_name: str = "Qwen/Qwen3-0.6B"
    enable_thinking: bool = False
    guided_json: bool = False
    tokenizer: AutoTokenizer = field(init=False)

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.enable_thinking:
            self.vllm_serve_args += ["--reasoning-parser", "deepseek_r1"]

    def get_state(self) -> dict:
        return {
            "model_name": self.model_name,
            "class_name": str(self.__class__.__name__),
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }

    def get_predictions(
        self, samples: list[dict], workers: int = 8
    ) -> tuple[list[dict], list[dict]]:
        self.max_concurrent_requests = workers
        return super().get_predictions(samples)

    async def async_get_predictions(
        self, samples: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        results: list[dict] = []

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
                    result, meta = await self.request_chat_completions(payload)
                result["completion"] = result
                result["meta"] = meta

                message = result["choices"][0]["message"]
                answer = message.get("content", "")
                reasoning = message.get("reasoning_content", "")

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

        errors = [result for result in results if result.get("error")]
        success_results = [result for result in results if not result.get("error")]

        logger.info(
            f"Obtained {len(success_results)} successful results and {len(errors)} errors"
        )
        return results
