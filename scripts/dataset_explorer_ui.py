from dotenv import load_dotenv
import streamlit as st
import json
from datasets import load_dataset
from collections import Counter
from typing import Optional
import os
from any2json.training.utils import load_hf_dataset
from any2json.utils import configure_loggers

st.set_page_config(
    page_title="Any2JSON Dataset Explorer", page_icon="🔍", layout="wide"
)


@st.cache_resource
def load_hf_dataset_cached(repo_id: str):
    return load_hf_dataset(repo_id)


@st.cache_data
def compute_metadata(_ds):
    content_types = []
    groups = []
    schema_ids = []
    synthetic_types = []
    parsed_metas = []

    for sample in _ds:
        meta = parse_meta(sample["meta"])
        parsed_metas.append(meta)
        content_types.append(meta.get("input_chunk_content_type", "unknown"))
        groups.append(meta.get("group"))
        schema_ids.append(meta.get("schema_id"))
        if meta.get("is_synthetic"):
            synthetic_types.append(meta.get("synthetic_type", "unknown"))

    return content_types, groups, schema_ids, synthetic_types, parsed_metas


def parse_meta(meta_str: str) -> dict:
    if isinstance(meta_str, str):
        return json.loads(meta_str)
    return meta_str


def parse_schema(schema_str: str) -> dict | str:
    if isinstance(schema_str, str):
        try:
            return json.loads(schema_str)
        except:
            return schema_str
    return schema_str


def parse_output(output_str: str) -> dict:
    if isinstance(output_str, str):
        return json.loads(output_str)
    return output_str


def main():
    st.title("🔍 Any2JSON Dataset Explorer")
    st.markdown(
        "Explore the training dataset for universal structured data to JSON conversion"
    )

    repo_id = st.sidebar.text_input(
        "HuggingFace Dataset ID", value="btseytlin/any2json"
    )

    with st.spinner("Loading dataset..."):
        try:
            dataset = load_hf_dataset_cached(repo_id)
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return

    st.sidebar.markdown("### Dataset Info")
    st.sidebar.metric("Train Samples", len(dataset["train"]))
    st.sidebar.metric("Test Samples", len(dataset["test"]))

    split = st.sidebar.selectbox("Split", ["train", "test"])
    ds = dataset[split]

    st.sidebar.markdown("---")

    content_types, groups, schema_ids, synthetic_types, parsed_metas = compute_metadata(
        ds
    )
    content_type_counts = Counter(content_types)
    group_counts = Counter(groups)

    st.sidebar.markdown("### Content Types")
    for ct, count in content_type_counts.most_common():
        st.sidebar.text(f"{ct}: {count}")

    st.sidebar.markdown("### Synthetic Samples")
    st.sidebar.metric("Total Synthetic", len(synthetic_types))
    for syn_type, count in Counter(synthetic_types).most_common():
        st.sidebar.text(f"{syn_type}: {count}")

    tab1, tab2, tab3 = st.tabs(
        ["🔎 Browse Samples", "📊 Statistics", "🎲 Random Sample"]
    )

    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            filter_content_type = st.multiselect(
                "Filter by Content Type",
                options=["all"] + list(content_type_counts.keys()),
                default=["all"],
            )

            filter_group = st.multiselect(
                "Filter by Group",
                options=["all"] + sorted([g for g in set(groups) if g is not None]),
                default=["all"],
            )

            filter_synthetic = st.selectbox(
                "Filter Synthetic", options=["all", "only_synthetic", "only_real"]
            )

        filtered_indices = []
        for idx in range(len(ds)):
            meta = parsed_metas[idx]

            if "all" not in filter_content_type:
                if meta.get("input_chunk_content_type") not in filter_content_type:
                    continue

            if "all" not in filter_group:
                if meta.get("group") not in filter_group:
                    continue

            if filter_synthetic == "only_synthetic" and not meta.get("is_synthetic"):
                continue
            if filter_synthetic == "only_real" and meta.get("is_synthetic"):
                continue

            filtered_indices.append(idx)

        st.markdown(f"**Showing {len(filtered_indices)} samples**")

        with col2:
            if filtered_indices:
                sample_idx = st.selectbox(
                    "Select Sample",
                    options=filtered_indices,
                    format_func=lambda x: f"Sample {x}",
                )

                sample = ds[sample_idx]
                meta = parsed_metas[sample_idx]

                st.markdown("### Sample Details")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Content Type", meta.get("input_chunk_content_type"))
                    st.metric("Group", meta.get("group", "N/A"))
                with col_b:
                    st.metric("Schema ID", meta.get("schema_id"))
                    st.metric("Synthetic", "Yes" if meta.get("is_synthetic") else "No")

                if meta.get("is_synthetic"):
                    st.info(f"Synthetic Type: {meta.get('synthetic_type', 'unknown')}")

                st.markdown("### Input Data")
                st.code(sample["input_data"], language="text")

                st.markdown("### Schema")
                schema = parse_schema(sample["schema"])
                if isinstance(schema, dict):
                    st.json(schema)
                else:
                    st.code(schema, language="text")

                st.markdown("### Output")
                output = parse_output(sample["output"])
                st.json(output)

                with st.expander("📋 Full Metadata"):
                    st.json(meta)
            else:
                st.info("No samples match the current filters")

    with tab2:
        st.markdown("### Dataset Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Content Type Distribution")
            st.bar_chart(dict(content_type_counts))

        with col2:
            st.markdown("#### Top Groups")
            top_groups = dict(group_counts.most_common(20))
            st.bar_chart(top_groups)

        st.markdown("#### Unique Schemas")
        st.metric("Number of Unique Schemas", len(set(schema_ids)))

        st.markdown("#### Synthetic vs Real")
        synthetic_count = sum(1 for s in synthetic_types if s)
        real_count = len(ds) - synthetic_count
        st.bar_chart({"Real": real_count, "Synthetic": synthetic_count})

    with tab3:
        if st.button("🎲 Get Random Sample"):
            import random

            random_idx = random.randint(0, len(ds) - 1)
            sample = ds[random_idx]
            meta = parsed_metas[random_idx]

            st.markdown(f"### Sample #{random_idx}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Content Type", meta.get("input_chunk_content_type"))
                st.metric("Group", meta.get("group", "N/A"))
            with col2:
                st.metric("Schema ID", meta.get("schema_id"))
                st.metric("Synthetic", "Yes" if meta.get("is_synthetic") else "No")

            st.markdown("### Input Data")
            st.code(sample["input_data"], language="text")

            st.markdown("### Schema")
            schema = parse_schema(sample["schema"])
            if isinstance(schema, dict):
                st.json(schema)
            else:
                st.code(schema, language="text")

            st.markdown("### Output")
            output = parse_output(sample["output"])
            st.json(output)

            with st.expander("📋 Full Metadata"):
                st.json(meta)


if __name__ == "__main__":
    load_dotenv(override=False)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    main()
