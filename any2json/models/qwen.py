from __future__ import annotations

import torch
from dataclasses import dataclass, field
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from any2json.utils import logger

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


@dataclass
class QwenModel:
    model_name: str | None = None
    enable_thinking: bool = False
    use_vllm: bool = True
    backend: str = "offline"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 8000
    impl: object = field(init=False)

    def __post_init__(self) -> None:
        if self.use_vllm and self.backend == "server":
            self.impl = QwenVLLMServer(
                model_name=self.model_name,
                enable_thinking=self.enable_thinking,
                max_tokens=self.max_tokens,
                base_url=self.base_url,
                api_key=self.api_key,
            )
        elif self.use_vllm:
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
                "use_vllm": self.use_vllm,
                "backend": self.backend,
            }
        )
        return state

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, dict]:
        return self.impl.convert_to_json(input_text, schema)
