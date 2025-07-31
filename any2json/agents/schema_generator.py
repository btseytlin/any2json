import asyncio
import json
import logging
import sys
from pydantic import Field, field_validator
import fastjsonschema
from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.fallback import FallbackModel
from any2json.utils import logger
from any2json.schema_utils import to_supported_json_schema
from pydantic_ai import Agent, RunContext
import tqdm.asyncio as tqdm_asyncio
import traceback

from tenacity import (
    before_sleep_log,
    retry,
    wait_exponential,
    stop_after_attempt,
)

AGENT_MAX_RETRIES = 4
AGENT_WAIT_MULTIPLIER = 3
AGENT_WAIT_MIN = 5
AGENT_WAIT_MAX = 180

SYSTEM_PROMPT = """
You are a JSONSchema generation expert. Your task is to:
1. Parse the provided JSON string
2. Analyze the structure and data types
3. Generate a comprehensive JSONSchema that describes the data
4. Ensure the schema is valid and follows JSON Schema specification
5. Use only the basic subset of JSON Schema. Only use the following types: string, number, boolean, array, object. No "format", "enum" or other attributes. 
No references and $defs. No "required", "additionalProperties", "minItems" or similar keys.
6. Make sure every type is nullable

Never output a "{{}}" or "true" as the schema. While they are valid, they are not useful. You can expect every string you get to have a realistic JSON schema.

Here is an example of an object and a matching valid JSONSchema:

Input:
{
    "name": "John Doe",
    "age": 25,
    "isStudent": true,
    "courses": ["Math", "Science"]
}

Output:
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
"""


class SchemaAgentInputSchema(BaseModel):
    """Input for JSONSchema generation."""

    input_json: dict | list | str | int | float | bool = Field(
        ...,
        description="JSON-able data to generate a JSONSchema for.",
    )
    previous_schema: dict | str | None = Field(
        None,
        description="The previous schema that was generated. If None, the first schema will be generated.",
    )
    error_message: str | None = Field(
        None,
        description="The error message that was returned by the previous schema validation. If None, the first schema will be generated.",
    )

    @field_validator("input_json")
    def validate_input_json(cls, v):
        return json.dumps(v)


class SchemaAgentOutputSchema(BaseModel):
    """Output of JSONSchema generation."""

    explanation: str = Field(
        ...,
        description="A short explanation of the schema generation process.",
    )
    output_schema: str = Field(
        ...,
        description="The generated JSONSchema as a valid JSONSchema json object, stringified.",
    )


class JSONSchemaGeneratorAgent:
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
            deps_type=SchemaAgentInputSchema,
            output_type=SchemaAgentOutputSchema,
        )

        @self.agent.system_prompt
        async def add_input(
            ctx: RunContext[SchemaAgentInputSchema],
        ) -> str:
            input_json = ctx.deps.input_json
            input_json_str = json.dumps(input_json, indent=1)
            prompt = f"""
            Generate a JSONSchema for this data:
            {input_json_str}
            """

            if ctx.deps.previous_schema and ctx.deps.error_message:
                prompt += f"""
                Previously you generated this schema:
                {json.dumps(ctx.deps.previous_schema, indent=1)}

                But it failed validation with the following error:
                {ctx.deps.error_message}

                Please correct the schema and return a valid JSONSchema.
                """

            return prompt

    @retry(
        stop=stop_after_attempt(AGENT_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=AGENT_WAIT_MULTIPLIER,
            min=AGENT_WAIT_MIN,
            max=AGENT_WAIT_MAX,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING, exc_info=True),
    )
    async def run_async(
        self,
        input_json: dict,
        previous_schema: dict | str | None = None,
        error_message: str | None = None,
    ) -> SchemaAgentOutputSchema:
        async with self.semaphore:
            logger.debug(
                f"Running schema generator agent for input_json: {input_json}, previous_schema: {previous_schema}, error_message: {error_message}"
            )

            return await self.agent.run(
                deps=SchemaAgentInputSchema(
                    input_json=input_json,
                    previous_schema=previous_schema,
                    error_message=error_message,
                ),
            )

    def validate_schema(
        self,
        generated_schema: dict,
        input_json: dict,
    ) -> bool:
        assert generated_schema, "Generated schema cannot be empty"
        logger.debug(f"Validating {input_json} against schema: {generated_schema}")

        validate = fastjsonschema.compile(generated_schema)
        validate(input_json)

    async def generate_and_validate_schema(self, input_json: dict) -> tuple[dict, str]:
        retries_used = 0
        error_message = ""
        previous_schema = None

        for _ in range(self.max_retries):
            try:
                result = await self.run_async(
                    input_json=input_json,
                    previous_schema=previous_schema,
                    error_message=error_message,
                )
                logger.debug(
                    "All messages: "
                    + json.dumps(json.loads(result.all_messages_json()), indent=1)
                )
                previous_schema = result.output.output_schema
                try:
                    generated_schema = json.loads(result.output.output_schema)
                    logger.debug(
                        f"Generated schema: {json.dumps(generated_schema, indent=1)}"
                    )
                except json.JSONDecodeError:
                    generated_schema = result.output.output_schema
                    logger.debug(f"Generated schema: {generated_schema}")
                    raise

                await asyncio.to_thread(
                    self.validate_schema,
                    generated_schema,
                    input_json,
                )

                logger.debug(f"Successfully validated")
                model_used = result.all_messages()[-2].model_name
                return generated_schema, model_used
            except Exception as e:
                retries_used += 1
                exc_info = sys.exc_info()

                error_message = "".join(
                    traceback.format_exception(exc_info[0], exc_info[1], exc_info[2])
                )
                logger.debug(f"Passing retry with error message: {error_message}")
                continue

        raise Exception(
            f"Failed to generate valid schema after {self.max_retries} attempts. Last error: {error_message}"
        )
