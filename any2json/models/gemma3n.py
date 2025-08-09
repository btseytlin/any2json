from __future__ import annotations

import json
import os
import torch
from any2json.utils import logger
from transformers import AutoProcessor
from transformers import Gemma3nForConditionalGeneration

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else device


class Gemma3nModel:
    system_prompt = """
    You are a helpful assistant that can convert structured data to JSON according to the provided JSONSchema.

    ## Task:
    Convert the input data to JSON according to the JSONSchema.
    If a property is not present in the input data, it should be present in the output and set to null.
    Ignore the "required" field in the JSONSchema.
    Return the resulting JSON object only, without any other text.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or "google/gemma-3n-E2B-it"
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, token=os.getenv("HF_TOKEN")
        )
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            token=os.getenv("HF_TOKEN"),
        ).eval()
        logger.info(f"Using device: {device}")
        self.model.to(device)

    def get_state(self) -> dict:
        return {
            "model_name": self.model_name,
            "class_name": str(self.__class__.__name__),
        }

    @classmethod
    def make_prompt(cls, input_text: str, schema: dict) -> str:
        return f"""
        ## Input data:
        {input_text}

        ## JSONSchema:
        {schema}
        """

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, dict | None]:
        prompt = self.make_prompt(input_text, schema)
        thinking_content, content = self.generate(prompt)
        return thinking_content, content

    def generate(self, prompt: str) -> tuple[str, dict | None]:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=32768, do_sample=False
            )
            generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True).strip(
            "\n"
        )
        return decoded, None
