import json
from typing import Any, Dict
import instructor
from pydantic import Field
import fastjsonschema
from pydantic_ai import Agent
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class SchemaAgentInputSchema(BaseModel):
    """Input for JSONSchema generation."""

    input_string: str = Field(
        ...,
        description="String containing JSON-able data to generate a JSONSchema for.",
    )


class SchemaAgentOutputSchema(BaseModel):
    """Output of JSONSchema generation."""

    explanation: str = Field(
        ...,
        description="A short explanation of the schema generation process.",
    )
    output_schema: str = Field(
        ...,
        description="The generated JSONSchema as a stringified json.",
    )


class JSONSchemaValidationAgent:
    def __init__(
        self,
        client: Any,
        model: str = "gemini-2.0-flash",
        max_retries: int = 3,
    ):
        system_prompt = """You are a JSONSchema generation expert. Your task is to:
1. Parse the provided JSON string
2. Analyze the structure and data types
3. Generate a comprehensive JSONSchema that describes the data
4. Ensure the schema is valid and follows JSON Schema specifications

Be precise with data types, required fields, and constraints. Generate schemas that are:
- Accurate to the actual data structure
- Comprehensive in type definitions
- Properly formatted according to JSON Schema standards
- Without descriptions
- Include appropriate constraints and validation rules

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
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
        },
        "age": {
            "type": "integer",
            "minimum": 0
        },
        "isStudent": {
            "type": "boolean",
        },
        "courses": {
            "type": "array",
            "items": {
                "type": "string"
            },
        }
    },
    "required": ["name", "age", "isStudent", "courses"]
}
"""

        self.agent = Agent(
            client=client,
            model=model,
            system_prompt=system_prompt,
            output_type=SchemaAgentOutputSchema,
        )
        self.max_retries = max_retries

    def generate_schema_prompt(
        self,
        input_data: str,
        error_message: str = "",
        previous_schema: dict[str, Any] | None = None,
    ) -> str:

        base_instruction = f"""Generate a JSONSchema for this data:
{input_data}

The schema must be valid and accurately describe the structure and types of the provided data.
Return only the JSONSchema as a valid JSON object.
"""
        if previous_schema:
            base_instruction += f"""

Previously you generated this schema:
{json.dumps(previous_schema, indent=2)}

But it failed validation with the following error:
{error_message}

Please correct the schema and return a valid JSONSchema.
"""
        return base_instruction

    def generate_schema(
        self,
        input_data: str,
        error_message: str = "",
        previous_schema: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        schema_prompt = self.generate_schema_prompt(
            input_data,
            error_message=error_message,
            previous_schema=previous_schema,
        )

        logger.debug(f"Prompt: {schema_prompt}")
        response = self.agent.run_sync(schema_prompt)

        logger.debug(f"Response: {response}")
        schema = json.loads(response.output.output_schema)
        return schema

    def validate_schema(
        self, generated_schema: Dict[str, Any], input_data: str
    ) -> bool:
        assert generated_schema, "Generated schema cannot be empty"
        logger.debug(f"Validating {input_data} against schema: {generated_schema}")
        validate = fastjsonschema.compile(generated_schema)
        validate(json.loads(input_data))

    def generate_and_validate_schema(self, input_data: SchemaAgentInputSchema) -> dict:
        retries_used = 0
        input_string = input_data.input_string
        error_message = ""
        previous_schema = None

        for _ in range(self.max_retries):
            try:
                logger.debug(
                    f"Generating schema for input: {input_string}, previous_schema: {previous_schema}, error_message: {error_message}"
                )
                generated_schema = self.generate_schema(
                    input_string, error_message, previous_schema
                )
                previous_schema = generated_schema

                self.validate_schema(generated_schema, input_string)

                logger.debug(f"Successfully validated schema: {generated_schema}")

                return generated_schema

            except (
                json.JSONDecodeError,
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
            f"Failed to generate valid schema after {self.max_retries} attempts. Last error: {error_message}"
        )
