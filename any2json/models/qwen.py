from __future__ import annotations
from tqdm import tqdm
import json
import torch
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from any2json.utils import logger


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
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, i, item): i for i, item in items}
        for f in as_completed(futs):
            i = futs[f]
            try:
                results.append((i, f.result()))
            except Exception as e:
                errors.append((i, e))
    results.sort(key=lambda x: x[0])
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
class QwenHF:
    model_name: str | None = None
    enable_thinking: bool = False
    max_tokens: int = 8000
    tokenizer: AutoTokenizer = field(init=False)
    model: AutoModelForCausalLM = field(init=False)

    def __post_init__(self) -> None:
        name = self.model_name or "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype="auto", device_map="auto"
        )
        logger.info(f"Using device: {device}")
        self.model.to(device)

    def get_state(self) -> dict:
        return {
            "model_name": self.model_name or "Qwen/Qwen3-0.6B",
            "class_name": str(self.__class__.__name__),
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
        }

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, dict]:
        prompt = make_prompt(input_text, schema)
        return self.generate(prompt)

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
        self, samples: list[dict], batch_size: int = 8
    ) -> tuple[list[dict], list[dict]]:
        results: list[dict] = []
        errors: list[dict] = []
        for start in tqdm(
            range(0, len(samples), batch_size),
            total=len(samples) // batch_size,
            desc="Generating predictions",
        ):
            batch = samples[start : start + batch_size]
            ids = list(range(start, start + len(batch)))
            texts = build_chat_texts(self.tokenizer, self.enable_thinking, batch)
            model_inputs = hf_tokenize_to_device(
                self.tokenizer, texts, self.model.device
            )
            output = self.model.generate(**model_inputs, max_new_tokens=self.max_tokens)
            input_lens = model_inputs.attention_mask.sum(dim=1).tolist()
            decoded = hf_decode_batch(self.tokenizer, output, input_lens)
            for j, idx in enumerate(ids):
                content, reasoning = parse_think(decoded[j])
                content = normalize_output_text(content)
                results.append(
                    {
                        "id": idx,
                        "answer": content,
                        "meta": {"thinking_content": reasoning},
                    }
                )
        return results, errors


@dataclass
class QwenVLLMBatch:
    model_name: str | None = None
    enable_thinking: bool = False
    max_tokens: int = 8000
    tokenizer: AutoTokenizer = field(init=False)
    vllm_llm: object = field(init=False)

    def __post_init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "vLLM offline backend requires CUDA. Use QwenVLLMServer with a running server."
            )
        name = self.model_name or "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        try:
            from vllm import LLM as VLLM
        except Exception as e:
            raise RuntimeError(f"Failed to import vLLM: {e}")
        self.vllm_llm = VLLM(model=name)

    def get_state(self) -> dict:
        return {
            "model_name": self.model_name or "Qwen/Qwen3-0.6B",
            "class_name": str(self.__class__.__name__),
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
        }

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, dict]:
        prompt = make_prompt(input_text, schema)
        return self.generate(prompt)

    def generate(self, prompt: str) -> tuple[str, dict]:
        try:
            from vllm import SamplingParams
        except Exception as e:
            raise RuntimeError(f"Failed to import vLLM SamplingParams: {e}")
        text = self.tokenizer.apply_chat_template(
            messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        temperature = 0.6 if self.enable_thinking else 0.7
        top_p = 0.95 if self.enable_thinking else 0.8
        params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=self.max_tokens
        )
        outputs = self.vllm_llm.generate([text], params)
        full_text = outputs[0].outputs[0].text
        content, reasoning = parse_think(full_text)
        content = normalize_output_text(content)
        return content, {"thinking_content": reasoning}

    def get_predictions(
        self, samples: list[dict], batch_size: int = 16
    ) -> tuple[list[dict], list[dict]]:
        try:
            params = vllm_sampling_params(self.enable_thinking, self.max_tokens)
        except Exception as e:
            raise RuntimeError(f"Failed to import vLLM SamplingParams: {e}")
        results: list[dict] = []
        errors: list[dict] = []
        for start in tqdm(
            range(0, len(samples), batch_size),
            total=len(samples) // batch_size,
            desc="Generating predictions",
        ):
            batch = samples[start : start + batch_size]
            ids = list(range(start, start + len(batch)))
            texts = build_chat_texts(self.tokenizer, self.enable_thinking, batch)
            outs = self.vllm_llm.generate(texts, params)
            for j, idx in enumerate(ids):
                t = outs[j].outputs[0].text
                content, reasoning = parse_think(t)
                content = normalize_output_text(content)
                results.append(
                    {
                        "id": idx,
                        "answer": content,
                        "meta": {"thinking_content": reasoning},
                    }
                )
        return results, errors


@dataclass
class QwenVLLMServer:
    model_name: str | None = None
    enable_thinking: bool = False
    max_tokens: int = 8000
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    tokenizer: AutoTokenizer = field(init=False)
    client: OpenAI = field(init=False)

    def __post_init__(self) -> None:
        name = self.model_name or "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def get_state(self) -> dict:
        return {
            "model_name": self.model_name or "Qwen/Qwen3-0.6B",
            "class_name": str(self.__class__.__name__),
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, dict]:
        prompt = make_prompt(input_text, schema)
        return self.generate(prompt)

    def generate(self, prompt: str) -> tuple[str, dict]:
        temperature = 0.6 if self.enable_thinking else 0.7
        top_p = 0.95 if self.enable_thinking else 0.8
        resp = self.client.chat.completions.create(
            model=self.model_name or "Qwen/Qwen3-0.6B",
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
        results: list[dict] = []
        errors: list[dict] = []

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

    def __post_init__(self) -> None:
        if self.backend == "vllm_server":
            self.impl = QwenVLLMServer(
                model_name=self.model_name,
                enable_thinking=self.enable_thinking,
                max_tokens=self.max_tokens,
                base_url=self.base_url,
                api_key=self.api_key,
            )
        elif self.backend == "vllm_offline":
            self.impl = QwenVLLMBatch(
                model_name=self.model_name,
                enable_thinking=self.enable_thinking,
                max_tokens=self.max_tokens,
            )
        else:
            self.impl = QwenHF(
                model_name=self.model_name,
                enable_thinking=self.enable_thinking,
                max_tokens=self.max_tokens,
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
