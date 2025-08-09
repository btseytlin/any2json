import json
import os
from google import genai

from any2json.schema_utils import to_supported_json_schema


class GeminiModel:
    system_prompt = """
    You are a helpful assistant that can convert structured data to JSON according to the provided JSONSchema.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.model = self.client.models.get(model=self.model_name)

    def get_state(self) -> dict:
        return {
            "model_name": self.model_name,
            "class_name": str(self.__class__.__name__),
        }

    def convert_to_json(self, input_text: str, schema: str) -> tuple[str, str]:
        prompt = f"""
        ## Input data:
        {input_text}

        ## JSONSchema:
        {schema}

        ## Task:
        Convert the input data to JSON according to the JSONSchema.
        If a property is not present in the input data, it should be present in the output and set to null.
        Ignore the "required" field in the JSONSchema.
        Return the resulting JSON object only, without any other text.
        """
        thinking_content, content = self.generate(prompt, schema)
        return thinking_content, content

    def generate(
        self, prompt: str, schema: dict | None = None
    ) -> tuple[str, None | dict]:
        full_prompt = f"{self.system_prompt}\n{prompt}"

        config = {
            "response_mime_type": "application/json",
        }

        # if schema:
        #     print(to_supported_json_schema(schema))
        #     config["response_schema"] = to_supported_json_schema(schema)

        response = self.client.models.generate_content(
            model=self.model.name,
            contents=full_prompt,
            config=config,
        )

        content = ""
        try:
            content = response.text
        except ValueError:
            content = f"Response was blocked: {response.prompt_feedback}"

        return content.strip(), None
