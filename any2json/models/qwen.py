import json
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenModel:
    system_prompt = """
    You are a helpful assistant that can convert structured data to JSON according to the provided JSONSchema.

    ## Task:
    Convert the input data to JSON according to the JSONSchema. 
    If a property is not present in the input data, it should be present in the output and set to null.
    Ignore the "required" field in the JSONSchema.
    Return the resulting JSON object only, without any other text.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

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

    def convert_to_json(self, input_text: str, schema: dict) -> tuple[str, str]:
        prompt = self.make_prompt(input_text, schema)
        thinking_content, content = self.generate(prompt)
        return thinking_content, content

    def generate(
        self, prompt: str, enable_thinking: bool = True
    ) -> tuple[str, None | dict]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        return content, {"thinking_content": thinking_content}
