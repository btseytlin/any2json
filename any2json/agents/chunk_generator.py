import asyncio
import json
import sys
from typing import Any, Dict
from pydantic import Field, field_validator
import fastjsonschema
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.fallback import FallbackModel
import traceback

from any2json.utils import logger

AGENT_MAX_RETRIES = 4

SYSTEM_PROMPT = """
Given a JsonSchema object, your task is to:
1. Inspect the provided JSON Schema.
2. Generate a plausible and non-trivial json that matches the schema,
3. In case of compilation errors, iteratively correct the json.

Avoid making obviously example data, generate realistic data. Never output things like "hello world", "example", "john doe", "sample", etc.

If schema has multiple fields of the same type, make sure the generated values are different an diverse.

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


class ChunkAgentInputSchema(BaseModel):
    """Input for JSON generation based on a JsonSchema."""

    input_schema: dict = Field(
        ...,
        description="JsonSchema to generate a matching JSON for.",
    )
    previous_json: dict | str | None = Field(
        None,
        description="The previous JSON that was generated. If None, the first JSON will be generated.",
    )
    error_message: str | None = Field(
        None,
        description="The error message that was returned by the previous JSON validation. If None, the first JSON will be generated.",
    )

    @field_validator("input_schema")
    def validate_input_schema(cls, v):
        return json.dumps(v) if isinstance(v, dict) else v


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
    system_prompt = SYSTEM_PROMPT

    def get_state(self) -> dict:
        return {
            "model_name": self.model_name,
            "enable_thinking": self.enable_thinking,
        }

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        fallback_model_name: str = "gemini-2.0-flash-lite",
        enable_thinking: bool = True,
        max_retries: int = 3,
        max_concurrent_tasks: int = 100,
    ):

        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

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

        self.fallback_model = None

        if fallback_model_name != model_name:
            self.fallback_model = GoogleModel(
                fallback_model_name,
                settings=GoogleModelSettings(
                    google_thinking_config=(
                        {
                            "thinking_budget": 1024,
                            "include_thoughts": False,
                        }
                    ),
                ),
            )

        self.agent = Agent(
            model=(
                FallbackModel(self.model, self.fallback_model)
                if self.fallback_model
                else self.model
            ),
            system_prompt=self.system_prompt,
            deps_type=ChunkAgentInputSchema,
            output_type=ChunkAgentOutputSchema,
            retries=AGENT_MAX_RETRIES,
        )

        @self.agent.system_prompt
        async def add_input(
            ctx: RunContext[ChunkAgentInputSchema],
        ) -> str:
            input_schema = ctx.deps.input_schema
            input_schema_str = (
                json.dumps(input_schema, indent=1)
                if isinstance(input_schema, dict)
                else input_schema
            )
            prompt = f"""
            Generate a JSON for this schema:
            {input_schema_str}
            """

            if ctx.deps.previous_json and ctx.deps.error_message:
                prompt += f"""
                Previously you generated this json:
                {json.dumps(ctx.deps.previous_json, indent=1)}

                But it failed validation with the following error:
                {ctx.deps.error_message}

                Please correct the json and return a valid JSON.
                """

            return prompt

    async def run_async(
        self,
        input_schema: dict,
        previous_json: dict | str | None = None,
        error_message: str | None = None,
    ) -> ChunkAgentOutputSchema:
        logger.debug(
            f"Running chunk generator agent for input_schema: {input_schema}, previous_json: {previous_json}, error_message: {error_message}"
        )

        return await self.agent.run(
            deps=ChunkAgentInputSchema(
                input_schema=input_schema,
                previous_json=previous_json,
                error_message=error_message,
            ),
        )

    def validate_json(
        self,
        generated_json: dict | list,
        input_schema: dict,
    ) -> bool:
        assert generated_json, "Generated json cannot be empty"
        logger.debug(f"Validating {generated_json} against schema: {input_schema}")

        validate = fastjsonschema.compile(input_schema)
        validate(generated_json)

    async def generate_and_validate_json(self, input_schema: dict) -> tuple[dict, str]:
        error_message = ""
        previous_json = None

        for i in range(self.max_retries):
            try:
                async with self.semaphore:
                    result = await self.run_async(
                        input_schema=input_schema,
                        previous_json=previous_json,
                        error_message=error_message,
                    )
                logger.debug(
                    "All messages: "
                    + json.dumps(json.loads(result.all_messages_json()), indent=1)
                )
                logger.debug(f"Generated json: {result.output.output_json}")
                previous_json = result.output.output_json
                generated_json = json.loads(result.output.output_json)

                await asyncio.to_thread(
                    self.validate_json,
                    generated_json,
                    input_schema,
                )

                logger.debug(f"Successfully validated")
                model_used = result.all_messages()[-2].model_name
                return generated_json, model_used
            except Exception as e:
                logger.debug(f"JSON generation attempt {i} failed")

                exc_info = sys.exc_info()

                error_message = "".join(
                    traceback.format_exception(exc_info[0], exc_info[1], exc_info[2])
                )
                continue

        raise Exception(
            f"Failed to generate after {self.max_retries} attempts. Last error: {error_message}"
        )
