import streamlit as st
import json
import httpx
from typing import Optional
import sys
import os

from any2json.schema_utils import to_supported_json_schema
from any2json.training.utils import format_example


def validate_json_schema(schema_text: str) -> tuple[bool, Optional[dict], str]:
    if not schema_text.strip():
        return True, None, ""

    try:
        schema = json.loads(schema_text)
        processed_schema = to_supported_json_schema(schema)
        return True, processed_schema, ""
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {e}"
    except ValueError as e:
        return False, None, f"Schema error: {e}"
    except Exception as e:
        return False, None, f"Unexpected error: {e}"


def call_vllm_inference(formatted_example: str, endpoint_url: str) -> tuple[bool, str]:
    try:
        payload = {
            "prompt": formatted_example,
            "max_tokens": 2048,
            "temperature": 0.1,
            "stop": ["[INPUT]", "[SCHEMA]"],
        }

        with httpx.Client() as client:
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
            is_valid, processed_schema, error_msg = validate_json_schema(schema_text)

            if not is_valid:
                st.error(f"Schema validation failed: {error_msg}")
                return

            schema_str = (
                json.dumps(processed_schema) if processed_schema else "[MISSING]"
            )
            formatted_example = format_example(input_data, schema_str)

            success, result = call_vllm_inference(formatted_example, endpoint_url)

            if success:
                with output_placeholder.container():
                    try:
                        parsed_json = json.loads(result.strip())
                        st.json(parsed_json)

                        with st.expander("Raw Output"):
                            st.code(result.strip(), language="json")

                    except json.JSONDecodeError:
                        st.warning("Output is not valid JSON, showing raw result:")
                        st.code(result.strip(), language="text")

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
        st.markdown("### 💡 Tips")
        st.markdown(
            """
        - Input data can be XML, CSV, YAML, or any structured format
        - Schema is optional but helps guide the conversion
        - The model will attempt to extract structured information
        - Make sure your vLLM server is running and accessible
        """
        )

        st.markdown("### 📊 Example Schema")
        example_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "emails": {"type": "array", "items": {"type": "string"}},
            },
        }
        st.json(example_schema)


if __name__ == "__main__":
    main()
