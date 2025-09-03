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
import traceback


AGENT_MAX_RETRIES = 4
# AGENT_WAIT_MULTIPLIER = 2
# AGENT_WAIT_MIN = 2
# AGENT_WAIT_MAX = 20

SYSTEM_PROMPT = """
You are a JSON-Schema specialist.  
Task: read the given JSON sample and output a JSON-Schema that:

1. Uses only these keywords: type, properties, items, $defs.  
2. Uses only these primitive types: string, number, integer, boolean, array, object, null.  
   • Every type MUST be expressed as an array that always includes "null".  
3. Never include: required, additionalProperties, enum, format, pattern, default, description, title, allOf, oneOf, anyOf, not, min*/max*, uniqueItems, dependent*.  
4. If a field is an object, describe its properties recursively with the same rules.  
5. If a field is an array, describe its items with the same rules.  
6. If the source has a "date" format, map it to type ["string","null"].  
7. Do NOT collapse the schema to {} or true; always emit a meaningful structure.
8. Make sure every type is nullable

Examples:

Example 1 - Basic object:
Input: {"name": "John", "age": 25, "isStudent": true, "courses": ["Math", "Science"]}
Output: {
    "type": ["object", "null"],
    "properties": {
        "name": {"type": ["string", "null"]},
        "age": {"type": ["integer", "null"]},
        "isStudent": {"type": ["boolean", "null"]},
        "courses": {
            "type": ["array", "null"],
            "items": {"type": ["string", "null"]}
        }
    }
}

Example 2 - Nested object:
Input: {"user": {"id": 123, "profile": {"email": "test@example.com", "score": 95.5}}}
Output: {
    "type": ["object", "null"],
    "properties": {
        "user": {
            "type": ["object", "null"],
            "properties": {
                "id": {"type": ["integer", "null"]},
                "profile": {
                    "type": ["object", "null"],
                    "properties": {
                        "email": {"type": ["string", "null"]},
                        "score": {"type": ["number", "null"]}
                    }
                }
            }
        }
    }
}

Example 3 - Array of objects:
Input: [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
Output: {
    "type": ["array", "null"],
    "items": {
        "type": ["object", "null"],
        "properties": {
            "id": {"type": ["integer", "null"]},
            "name": {"type": ["string", "null"]}
        }
    }
}

Example 4 - Mixed types array:
Input: {"data": [42, "text", true, null]}
Output: {
    "type": ["object", "null"],
    "properties": {
        "data": {
            "type": ["array", "null"],
            "items": {"type": ["integer", "string", "boolean", "null"]}
        }
    }
}

Return only the JSON-Schema, nothing else.
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
        timeout: int = 30,
    ):

        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.timeout = timeout

        self.model = GoogleModel(
            model_name,
            settings=(
                GoogleModelSettings(
                    timeout=self.timeout,
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
                    timeout=self.timeout,
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
            retries=AGENT_MAX_RETRIES,
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
                prev_str = (
                    json.dumps(ctx.deps.previous_schema, indent=1)
                    if isinstance(ctx.deps.previous_schema, dict)
                    else ctx.deps.previous_schema
                )

                prompt += f"""
                Previously you generated this schema:
                {prev_str}

                But it failed validation with the following error:
                {ctx.deps.error_message}

                Please correct the schema and return a valid JSONSchema.
                """

            return prompt

    async def run_async(
        self,
        input_json: dict,
        previous_schema: dict | str | None = None,
        error_message: str | None = None,
    ) -> SchemaAgentOutputSchema:
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
        error_message = ""
        previous_schema = None

        for i in range(self.max_retries):
            try:
                async with self.semaphore:
                    result = await self.run_async(
                        input_json=input_json,
                        previous_schema=previous_schema,
                        error_message=error_message,
                    )
                logger.debug(
                    "All messages: "
                    + json.dumps(json.loads(result.all_messages_json()), indent=1)
                )
                logger.debug(f"Generated schema: {result.output.output_schema}")
                previous_schema = result.output.output_schema
                generated_schema = json.loads(result.output.output_schema)
                previous_schema = generated_schema

                await asyncio.to_thread(
                    self.validate_schema,
                    generated_schema,
                    input_json,
                )

                logger.debug(f"Successfully validated")
                model_used = result.all_messages()[-2].model_name
                return generated_schema, model_used
            except Exception as e:
                logger.debug(f"Schema generation attempt {i} failed")

                exc_info = sys.exc_info()

                error_message = "".join(
                    traceback.format_exception(exc_info[0], exc_info[1], exc_info[2])
                )
                continue

        raise Exception(
            f"Failed to generate after {self.max_retries} attempts. Last error: {error_message}"
        )
