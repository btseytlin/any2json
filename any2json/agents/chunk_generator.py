import json
from typing import Any, Dict
import instructor
from pydantic import Field
import fastjsonschema
from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

from any2json.utils import logger
from any2json.schema_utils import to_supported_json_schema


class ChunkAgentInputSchema(BaseModel):
    """Input for JSON generation based on a JsonSchema."""

    input_schema: dict = Field(
        ...,
        description="JsonSchema to generate a matching JSON for.",
    )


class ChunkAgentOutputSchema(BaseModel):
    """Output of JSON generation."""

    explanation: str = Field(
        ...,
        description="A short explanation of the json generation process.",
    )
    output_json: str = Field(
        ...,
        description="The generated JSON as a stringified json.",
    )


class JSONChunkGeneratorAgent:
    def get_state(self) -> dict:
        return {
            "model_name": self.model_name,
            "enable_thinking": self.enable_thinking,
        }

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        max_retries: int = 3,
        enable_thinking: bool = True,
    ):
        self.system_prompt = """Given a JsonSchema object, your task is to:
1. Inspect the provided JSON Schema.
2. Generate a plausible and non-trivial json that matches the schema,
3. In case of compilation errors, iteratively correct the json,

Avoid making obviously example data, generate realistic data. Avoid things like "hello world", "example", "john doe", etc.

Here is an example of a valid JSONSchema and a matching valid JSON:

Input:
{
    "type": ["object", "null"],
    "properties": {
        "name": {
            "type": ["string", "null"],
        },
        "age": {
            "type": ["integer", "null"],
        },
        "isStudent": {
            "type": ["boolean", "null"],
        },
        "courses": {
            "type": ["array", "null"],
            "items": {
                "type": ["string", "null"],
            },
        }
    },
}

Output:
{
    "name": "Sam Smith",
    "age": 25,
    "isStudent": true,
    "courses": ["Math", "Science"]
}

"""
        self.model_name = model_name

        self.model = GoogleModel(
            model_name,
            settings=(
                GoogleModelSettings(
                    google_thinking_config=(
                        {
                            "thinking_budget": 1024,
                            "include_thoughts": False,
                        }
                    ),
                )
                if enable_thinking
                else None
            ),
        )

        self.agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            output_type=ChunkAgentOutputSchema,
        )
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking

    def generate_json_prompt(
        self,
        input_schema: str,
        error_message: str = "",
        previous_json: dict[str, Any] | None = None,
    ) -> str:

        base_instruction = f"""Generate a JSON for this schema:
{input_schema}

The json must be valid and match the provided schema while not being trivial.
"""
        if previous_json:
            base_instruction += f"""

Previously you generated this json:
{json.dumps(previous_json, indent=2)}

But it failed validation with the following error:
{error_message}

Please correct the json.
"""
        return base_instruction

    def generate_json(
        self,
        input_schema: str,
        error_message: str = "",
        previous_json: str | None = None,
    ) -> str:
        json_prompt = self.generate_json_prompt(
            input_schema,
            error_message=error_message,
            previous_json=previous_json,
        )

        logger.debug(f"LLM prompt: {json_prompt}")
        response = self.agent.run_sync(json_prompt)

        logger.debug(f"LLM response: {response}")
        generated_json = json.loads(response.output.output_json)
        return generated_json

    def validate_json(
        self,
        generated_json: dict | list,
        input_schema: dict,
    ) -> bool:
        assert generated_json, "Generated json cannot be empty"
        logger.debug(
            f"Validating generated {generated_json} against schema {input_schema}"
        )
        validate = fastjsonschema.compile(input_schema)
        validate(generated_json)

    def generate_and_validate_json(self, input: ChunkAgentInputSchema) -> dict:
        retries_used = 0
        input_schema = input.input_schema
        error_message = ""
        previous_json = None

        for _ in range(self.max_retries):
            try:
                logger.debug(
                    f"Generating json for input: {input_schema}, previous_json: {previous_json}, error_message: {error_message}"
                )
                generated_json = self.generate_json(
                    input_schema, error_message, previous_json
                )
                if isinstance(generated_json, str):
                    generated_json = json.loads(generated_json)
                previous_json = generated_json
                self.validate_json(generated_json, input_schema)

                logger.debug(f"Successfully validated json: {generated_json}")

                return generated_json

            except (
                json.JSONDecodeError,
                json.decoder.JSONDecodeError,
                fastjsonschema.JsonSchemaException,
                AssertionError,
            ) as e:
                logger.debug(e, exc_info=True)
                retries_used += 1
                error_message = (
                    f"repr(e): {repr(e)}, str(e): {str(e)}, type(e): {type(e)}"
                )
                continue

        raise Exception(
            f"Failed to generate after {self.max_retries} attempts. Last error: {error_message}"
        )
