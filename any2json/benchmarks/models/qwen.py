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
class BaseQwen:
    model_name: str | None = None
    enable_thinking: bool = False
    max_tokens: int = 8000
    batch_size: int = 16

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or "Qwen/Qwen3-0.6B"

    def get_state(self) -> dict:
        return {
            "model_name": self.resolved_model_name,
            "class_name": str(self.__class__.__name__),
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
        }

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, dict]:
        prompt = make_prompt(input_text, schema)
        return self.generate(prompt)

    def to_answer_meta(self, text: str) -> tuple[str, dict]:
        content, reasoning = parse_think(text)
        content = normalize_output_text(content)
        return content, {"thinking_content": reasoning}

    def iter_batches(
        self, samples: list[dict], batch_size: int
    ) -> Iterator[tuple[list[int], list[dict]]]:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            ids = list(range(start, start + len(batch)))
            yield ids, batch

    def collect_batch_outputs(
        self,
        samples: list[dict],
        batch_size: int,
        run_batch_fn: Callable[[list[dict]], tuple[object, Any]],
    ) -> tuple[list[object], list[Any], list[list[int]]]:
        outputs: list[object] = []
        aux_list: list[Any] = []
        id_list: list[list[int]] = []
        pbar = tqdm(total=len(samples), desc="Generating predictions")
        for ids, batch in self.iter_batches(samples, batch_size):
            try:
                out, aux = run_batch_fn(batch)
                outputs.append(out)
                aux_list.append(aux)
                id_list.append(ids)
                pbar.update(len(batch))
            except KeyboardInterrupt:
                logger.error("Keyboard interrupt")
                break
        return outputs, aux_list, id_list

    def decode_and_build_results(
        self,
        outputs: list[object],
        aux_list: list[Any],
        id_list: list[list[int]],
        decode_batch_fn: Callable[[object, Any], list[str]],
    ) -> list[dict]:
        results: list[dict] = []
        for out, aux, ids in tqdm(
            zip(outputs, aux_list, id_list, strict=True),
            total=len(outputs),
            desc="Decoding outputs",
        ):
            texts = decode_batch_fn(out, aux)
            ms_list = []
            if isinstance(aux, dict) and "inference_ms_list" in aux:
                ms_list = aux["inference_ms_list"]
            else:
                ms_list = [None] * len(ids)
            for j, idx in enumerate(ids):
                content, meta = self.to_answer_meta(texts[j])
                if j < len(ms_list) and ms_list[j] is not None:
                    meta.update({"inference_ms": ms_list[j]})
                results.append({"id": idx, "answer": content, "meta": meta})
        return results

    def run_batched_inference(
        self,
        samples: list[dict],
        batch_size: int,
        run_batch_fn: Callable[[list[dict]], tuple[object, Any]],
        decode_batch_fn: Callable[[object, Any], list[str]],
    ) -> tuple[list[dict], list[dict]]:
        outputs, aux_list, id_list = self.collect_batch_outputs(
            samples, batch_size, run_batch_fn
        )
        results = self.decode_and_build_results(
            outputs, aux_list, id_list, decode_batch_fn
        )
        return results, []


@dataclass
class QwenVLLMServer(VLLMServerModel):
    enable_thinking: bool = False
    batch_size: int = 16
    tokenizer: AutoTokenizer = field(init=False)

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.resolved_model_name)
        if self.enable_thinking:
            self.vllm_serve_args += ["--reasoning-parser", "deepseek_r1"]

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or "Qwen/Qwen3-0.6B"

    def get_state(self) -> dict:
        return {
            "model_name": self.resolved_model_name,
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
            async with semaphore:
                x = to_text(sample["input_data"])
                prompt = make_prompt(x, sample["schema"])

                params = get_sampling_params(self.enable_thinking, self.max_tokens)
                payload = {
                    "model": self.resolved_model_name,
                    "messages": messages(prompt),
                    "max_tokens": params["max_tokens"],
                    "temperature": params["temperature"],
                    "top_p": params["top_p"],
                    "extra_body": {
                        "top_k": params["top_k"],
                        "min_p": params["min_p"],
                    },
                }

                try:
                    result, meta = await self.request_chat_completions(payload)
                    message = result["choices"][0]["message"]
                    answer = message.get("content", "")
                    reasoning = message.get("reasoning_content", "")

                    if not reasoning:
                        answer, reasoning = parse_think(answer)

                    content = normalize_output_text(answer)
                    meta.update({"thinking_content": reasoning})
                    return {
                        "id": i,
                        "answer": content,
                        "meta": meta,
                    }
                except Exception as e:
                    logger.error(e, exc_info=True)
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback_str = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    )
                    return {
                        "id": i,
                        "error": str(e),
                        "traceback": traceback_str,
                    }

        tasks = [task(i, sample) for i, sample in enumerate(samples)]
        results = await tqdm_asyncio.gather(*tasks, desc="Executing requests")

        errors = [result for result in results if "error" in result]
        success_results = [result for result in results if "error" not in result]

        logger.info(
            f"Obtained {len(success_results)} successful results and {len(errors)} errors"
        )
        return success_results, errors
