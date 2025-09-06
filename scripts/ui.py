from collections.abc import Callable
import streamlit as st
import json
import httpx
from typing import Optional
import sys
import os

from any2json.schema_utils import to_supported_json_schema
from any2json.training.utils import format_example
import fastjsonschema

from any2json.utils import json_dumps_minified


def validate_json_schema(
    schema_text: str,
) -> tuple[bool, Optional[dict], str, Optional[Callable]]:
    if not schema_text.strip():
        return True, None, None, "", None

    try:
        schema = json.loads(schema_text)
        processed_schema = to_supported_json_schema(schema)
        validator = fastjsonschema.compile(processed_schema)
        return True, processed_schema, schema, "", validator
    except json.JSONDecodeError as e:
        return False, None, None, f"Invalid JSON: {e}", None
    except ValueError as e:
        return False, None, None, f"Schema error: {e}", None
    except Exception as e:
        return False, None, None, f"Unexpected error: {e}", None


def call_vllm_inference(
    formatted_example: str,
    endpoint_url: str,
    json_schema: dict | None = None,
) -> tuple[bool, str]:
    try:
        payload = {
            "prompt": formatted_example,
            "max_tokens": 1024,
            "temperature": 0,
        }

        if json_schema:
            payload["guided_json"] = json_schema

        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{endpoint_url}/v1/completions",
                json=payload,
            )

        if response.status_code == 200:
            result = response.json()
            completion = result["choices"][0]["text"]
            return True, completion
        else:
            return False, f"HTTP {response.status_code}: {response.text}"

    except httpx.RequestError as e:
        return False, f"Request error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def main():
    st.set_page_config(page_title="Any2JSON Inference", page_icon="🔄", layout="wide")

    st.title("🔄 Any2JSON Model Inference")
    st.markdown("Convert any structured data to JSON using our trained model")

    st.subheader("⚙️ Configuration")
    endpoint_url = st.text_input(
        "vLLM Endpoint URL",
        value="http://localhost:8000",
        help="URL of your vLLM inference server",
    )
    enable_guided_json = st.checkbox("Enable Guided Decoding", value=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Input")

        input_data = st.text_area(
            "Input Data",
            height=300,
            placeholder="Enter your data here (XML, CSV, etc.)",
            help="The structured data you want to convert to JSON",
        )

        schema_text = st.text_area(
            "JSON Schema (Optional)",
            height=200,
            placeholder='{"type": "object", "properties": {...}}',
            help="Optional JSON schema to guide the conversion",
        )

    with col2:
        st.subheader("✅ Output")

        output_placeholder = st.empty()
        with output_placeholder.container():
            st.info("Results will appear here after conversion")

    if st.button("🚀 Convert to JSON", type="primary", use_container_width=True):
        if not input_data.strip():
            st.error("Please provide input data")
            return

        with st.spinner("Processing..."):
            is_valid, processed_schema, original_schema, error_msg, validator = (
                validate_json_schema(schema_text)
            )

            if not is_valid:
                st.error(f"Schema validation failed: {error_msg}")
                return

            schema_str = (
                json_dumps_minified(processed_schema)
                if processed_schema
                else "[MISSING]"
            )
            formatted_example = format_example(input_data, schema_str)

            success, result = call_vllm_inference(
                formatted_example,
                endpoint_url,
                json_schema=(
                    original_schema if original_schema and enable_guided_json else None
                ),
            )
            if success:
                with output_placeholder.container():
                    json_parsed = False
                    schema_validation_passed = False
                    try:
                        parsed_json = json.loads(result.strip())
                        json_parsed = True

                        if validator:
                            try:
                                print("Validating", parsed_json)
                                val_result = validator(parsed_json)
                                print("Validation result", val_result)
                                schema_validation_passed = True
                            except Exception as e:
                                pass

                    except json.JSONDecodeError:
                        st.warning("Output is not valid JSON, showing raw result:")
                        st.code(result.strip(), language="text")

                    if json_parsed:
                        st.success("✅ Valid JSON")

                        if validator:
                            if schema_validation_passed:
                                st.success("✅ Schema validation passed")
                            else:
                                st.warning("❌ Schema validation failed")

                        st.json(parsed_json)

                    else:
                        st.warning("❌ Invalid JSON")

                    with st.expander("Raw Output"):
                        st.code(result.strip(), language="json")

            else:
                with output_placeholder.container():
                    st.error(f"Inference failed: {result}")

            if processed_schema:
                st.subheader("🔧 Processed Schema")
                with st.expander("View processed schema details", expanded=False):
                    st.markdown(
                        "**Original schema was processed to ensure compatibility with the model**"
                    )
                    st.code(json.dumps(processed_schema, indent=2), language="json")

            st.subheader("🔧 Formatted Example")
            with st.expander("View formatted prompt", expanded=False):
                st.code(formatted_example, language="text")

    with st.sidebar:
        st.markdown("### Example input (TOML)")
        st.code(
            """
[[films]]
title = "Memento"
release_year = 2000
budget = 10000000
worldwide_gross = 46249333

[[films]]
title = "Insomnia"
release_year = 2002
budget = 58000000
worldwide_gross = 113494737

[[films]]
title = "The Dark Knight"
release_year = 2008
budget = 185000000
worldwide_gross = 1005197135


        """
        )

        st.markdown("### 📊 Example Schema")
        example_schema = {
            "type": "object",
            "properties": {
                "films": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "release_year": {"type": "number"},
                            "budget": {"type": "number"},
                            "worldwide_gross": {"type": "number"},
                        },
                    },
                }
            },
        }
        st.json(example_schema)


if __name__ == "__main__":
    main()
