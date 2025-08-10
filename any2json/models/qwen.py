from __future__ import annotations
from tqdm import tqdm
import json
import torch
from dataclasses import dataclass, field
from typing import Iterator, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from any2json.utils import logger
import subprocess
import sys
import time
import httpx
from urllib.parse import urlparse


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


def vllm_sampling_params(enable_thinking: bool, max_tokens: int):
    from vllm import SamplingParams

    temperature = 0.6 if enable_thinking else 0.7
    top_p = 0.95 if enable_thinking else 0.8
    return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)


def parallel_map(
    fn, items: list, max_workers: int
) -> tuple[list[tuple[int, object]], list[tuple[int, Exception]]]:
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
                    errors.append((i, e))
                pbar.update(1)

            except KeyboardInterrupt:
                logger.error("Keyboard interrupt")
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
            for j, idx in enumerate(ids):
                content, meta = self.to_answer_meta(texts[j])
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
class QwenHF(BaseQwen):
    model_name: str | None = None
    enable_thinking: bool = False
    max_tokens: int = 8000
    tokenizer: AutoTokenizer = field(init=False)
    model: AutoModelForCausalLM = field(init=False)
    batch_size: int = 16

    def __post_init__(self) -> None:
        name = self.resolved_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype="auto", device_map="auto"
        )
        self.batch_size = self.batch_size or 16
        logger.info(f"Using device: {device}")
        self.model.to(device)
        self.model.eval()
        self.model = torch.compile(self.model)

    def get_state(self) -> dict:
        return super().get_state()

    def generate(self, prompt: str) -> tuple[str, dict]:
        text = self.tokenizer.apply_chat_template(
            messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")
        model_inputs = model_inputs.to(self.model.device)
        output = self.model.generate(**model_inputs, max_new_tokens=self.max_tokens)
        full_text = self.tokenizer.decode(
            output[0][len(model_inputs.input_ids[0]) :], skip_special_tokens=True
        ).strip()
        content, reasoning = parse_think(full_text)
        content = normalize_output_text(content)
        return content, {"thinking_content": reasoning}

    def get_predictions(
        self, samples: list[dict], batch_size: int | None = None
    ) -> tuple[list[dict], list[dict]]:
        batch_size = batch_size or self.batch_size

        def run_batch_fn(batch: list[dict]) -> tuple[object, list[int]]:
            texts = build_chat_texts(self.tokenizer, self.enable_thinking, batch)
            model_inputs = hf_tokenize_to_device(
                self.tokenizer, texts, self.model.device
            )
            output = self.model.generate(**model_inputs, max_new_tokens=self.max_tokens)
            input_lens = model_inputs.attention_mask.sum(dim=1).tolist()
            return output, input_lens

        def decode_batch_fn(output: object, input_lens: list[int]) -> list[str]:
            return hf_decode_batch(self.tokenizer, output, input_lens)

        return self.run_batched_inference(
            samples, batch_size, run_batch_fn, decode_batch_fn
        )


@dataclass
class QwenVLLMBatch(BaseQwen):
    model_name: str | None = None
    enable_thinking: bool = False
    max_tokens: int = 8000
    tokenizer: AutoTokenizer = field(init=False)
    vllm_llm: object = field(init=False)
    batch_size: int = 16

    def __post_init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "vLLM offline backend requires CUDA. Use QwenVLLMServer with a running server."
            )
        name = self.resolved_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        try:
            from vllm import LLM as VLLM
        except Exception as e:
            raise RuntimeError(f"Failed to import vLLM: {e}")
        self.vllm_llm = VLLM(model=name)

    def get_state(self) -> dict:
        return super().get_state()

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, dict]:
        return super().convert_to_json(input_text, schema)

    def generate(self, prompt: str) -> tuple[str, dict]:
        text = self.tokenizer.apply_chat_template(
            messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        params = vllm_sampling_params(self.enable_thinking, self.max_tokens)
        outputs = self.vllm_llm.generate([text], params)
        full_text = outputs[0].outputs[0].text
        content, reasoning = parse_think(full_text)
        content = normalize_output_text(content)
        return content, {"thinking_content": reasoning}

    def get_predictions(
        self, samples: list[dict], batch_size: int | None = None
    ) -> tuple[list[dict], list[dict]]:
        batch_size = batch_size or self.batch_size
        params = vllm_sampling_params(self.enable_thinking, self.max_tokens)

        def run_batch_fn(batch: list[dict]) -> tuple[list[object], None]:
            texts = build_chat_texts(self.tokenizer, self.enable_thinking, batch)
            outs = self.vllm_llm.generate(texts, params)
            return outs, None

        def decode_batch_fn(outs: list[object], _: None) -> list[str]:
            return [o.outputs[0].text for o in outs]

        return self.run_batched_inference(
            samples, batch_size, run_batch_fn, decode_batch_fn
        )


@dataclass
class QwenVLLMServer(BaseQwen):
    model_name: str | None = None
    enable_thinking: bool = False
    max_tokens: int = 8000
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    tokenizer: AutoTokenizer = field(init=False)
    client: OpenAI = field(init=False)
    batch_size: int = 16
    server_process: subprocess.Popen | None = field(default=None, init=False)
    server_startup_timeout: float = 120.0

    def __post_init__(self) -> None:
        name = self.resolved_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def parse_host_port(self) -> tuple[str, int]:
        u = urlparse(self.base_url)
        return (u.hostname or "127.0.0.1", u.port or 8000)

    def health_url(self) -> str:
        return f"{self.base_url}/models"

    def is_server_alive(self) -> bool:
        try:
            r = httpx.get(self.health_url(), timeout=2.0)
            logger.info(f"Health check response: {r.status_code} {r.text}")
            return r.status_code == 200
        except Exception:
            return False

    def build_server_command(self) -> list[str]:
        host, port = self.parse_host_port()

        args = []
        if self.enable_thinking:
            args.append("--enable-reasoning")
            args += ["--reasoning-parser", "deepseek_r1"]
        if self.max_tokens:
            args += ["--max-model-len", str(self.max_tokens)]

        return [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.resolved_model_name,
            "--host",
            host,
            "--port",
            str(port),
            *args,
        ]

    def spawn_server(self, cmd: list[str]) -> None:
        try:
            self.server_process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start vLLM server: {e}")

    def wait_for_server_ready(self) -> None:
        deadline = time.time() + self.server_startup_timeout
        while time.time() < deadline:
            if self.is_server_alive():
                return
            if self.server_process and self.server_process.poll() is not None:
                raise RuntimeError("vLLM server exited early")
            time.sleep(0.5)
        raise TimeoutError("Timed out waiting for vLLM server to start")

    def ensure_server_started(self) -> bool:
        logger.info("Checking if server is alive")
        if self.is_server_alive():
            logger.warning("VLLM server is already running")
            raise RuntimeError("VLLM server is already running")
        cmd = self.build_server_command()
        logger.info(f"Starting server with command: {cmd}")
        self.spawn_server(cmd)
        logger.info("Waiting for server to be ready")
        self.wait_for_server_ready()
        logger.info("VLLM server is ready")
        return True

    def stop_server(self) -> None:
        if self.server_process is None:
            return
        try:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        finally:
            self.server_process = None

    def get_state(self) -> dict:
        s = super().get_state()
        s.update({"base_url": self.base_url})
        return s

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, dict]:
        prompt = make_prompt(input_text, schema)
        return self.generate(prompt)

    def generate(self, prompt: str) -> tuple[str, dict]:
        temperature = 0.6 if self.enable_thinking else 0.7
        top_p = 0.95 if self.enable_thinking else 0.8
        resp = self.client.chat.completions.create(
            model=self.resolved_model_name,
            messages=messages(prompt),
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.max_tokens,
        )
        m = resp.choices[0].message
        content = m.content or ""
        reasoning = getattr(m, "reasoning_content", "") or ""
        if not reasoning:
            content, reasoning = parse_think(content)
        content = normalize_output_text(content)
        return content, {"thinking_content": reasoning}

    def get_predictions(
        self, samples: list[dict], workers: int = 8
    ) -> tuple[list[dict], list[dict]]:
        started = False
        try:
            started = self.ensure_server_started()
            return self.execute_parallel_requests(samples, workers)
        finally:
            if started:
                self.stop_server()

    def execute_parallel_requests(
        self, samples: list[dict], workers: int
    ) -> tuple[list[dict], list[dict]]:
        results: list[dict] = []
        errors: list[dict] = []

        logger.info(
            f"Executing {len(samples)} requests in parallel with {workers} workers"
        )

        def task(_: int, s: dict) -> tuple[str, dict]:
            x = to_text(s["input_data"])
            prompt = make_prompt(x, s["schema"])
            return self.generate(prompt)

        items = [(i, s) for i, s in enumerate(samples)]
        ok, err = parallel_map(task, items, workers)
        for i, (a, m) in ok:
            results.append({"id": i, "answer": a, "meta": m})
        for i, e in err:
            errors.append({"id": i, "error": str(e)})
        results.sort(key=lambda x: x["id"])
        return results, errors


@dataclass
class QwenModel:
    model_name: str | None = None
    enable_thinking: bool = False
    backend: str = "torch"  # vllm_offline, vllm_server, torch
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 8000
    impl: object = field(init=False)
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.backend == "vllm_server":
            self.impl = QwenVLLMServer(
                model_name=self.model_name,
                enable_thinking=self.enable_thinking,
                max_tokens=self.max_tokens,
                base_url=self.base_url,
                api_key=self.api_key,
                batch_size=self.batch_size,
            )
        elif self.backend == "vllm_offline":
            self.impl = QwenVLLMBatch(
                model_name=self.model_name,
                enable_thinking=self.enable_thinking,
                max_tokens=self.max_tokens,
                batch_size=self.batch_size,
            )
        else:
            self.impl = QwenHF(
                model_name=self.model_name,
                enable_thinking=self.enable_thinking,
                max_tokens=self.max_tokens,
                batch_size=self.batch_size,
            )

    def get_state(self) -> dict:
        state = self.impl.get_state()
        state.update(
            {
                "wrapper": "QwenModel",
                "backend": self.backend,
            }
        )
        return state

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, dict]:
        return self.impl.convert_to_json(input_text, schema)

    def get_predictions(
        self, samples: list[dict], **kwargs
    ) -> tuple[list[dict], list[dict]]:
        return self.impl.get_predictions(samples, **kwargs)
