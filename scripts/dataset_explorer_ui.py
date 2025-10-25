from dotenv import load_dotenv
import streamlit as st
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from any2json.database.client import get_db_session
from any2json.database.models import Chunk, JsonSchema, SchemaConversion
from any2json.utils import configure_loggers
from sqlalchemy import func, select

st.set_page_config(
    page_title="Any2JSON Dataset Explorer", page_icon="🔍", layout="wide"
)


@dataclass
class Sample:
    id: int
    input_content: str
    input_content_type: str
    schema_content: dict
    output_content: str
    is_synthetic: bool
    schema_id: int
    input_chunk_id: int
    output_chunk_id: int
    meta: dict | None


@st.cache_resource
def get_session(db_path: str):
    return get_db_session(f"sqlite:///{db_path}")


@st.cache_data
def load_samples(_session) -> list[Sample]:
    conversions = _session.query(SchemaConversion).order_by(SchemaConversion.id).all()

    samples = []
    for conv in conversions:
        sample = Sample(
            id=conv.id,
            input_content=conv.input_chunk.content,
            input_content_type=conv.input_chunk.content_type,
            schema_content=conv.schema.content,
            output_content=conv.output_chunk.content,
            is_synthetic=conv.input_chunk.is_synthetic,
            schema_id=conv.schema_id,
            input_chunk_id=conv.input_chunk_id,
            output_chunk_id=conv.output_chunk_id,
            meta=conv.input_chunk.meta or {},
        )
        samples.append(sample)

    return samples


@st.cache_data
def compute_stats(samples: list[Sample]):
    content_types = [s.input_content_type for s in samples]
    groups = [s.meta.get("group") for s in samples if s.meta]
    schema_ids = [s.schema_id for s in samples]
    synthetic_flags = [s.is_synthetic for s in samples]
    synthetic_types = [
        s.meta.get("synthetic_type") for s in samples if s.is_synthetic and s.meta
    ]

    return content_types, groups, schema_ids, synthetic_flags, synthetic_types


def main():
    st.title("🔍 Any2JSON Dataset Explorer")
    st.markdown(
        "Explore the training dataset for universal structured data to JSON conversion"
    )

    db_path = st.sidebar.text_input("Database Path", value="data/database.db")

    if not Path(db_path).exists():
        st.error(f"Database not found at {db_path}")
        return

    with st.spinner("Loading samples from database..."):
        try:
            session = get_session(db_path)
            samples = load_samples(session)
        except Exception as e:
            st.error(f"Failed to load samples: {e}")
            return

    st.sidebar.markdown("### Dataset Info")
    st.sidebar.metric("Total Samples", len(samples))

    st.sidebar.markdown("---")

    content_types, groups, schema_ids, synthetic_flags, synthetic_types = compute_stats(
        samples
    )
    content_type_counts = Counter(content_types)
    group_counts = Counter([g for g in groups if g is not None])

    st.sidebar.markdown("### Content Types")
    for ct, count in content_type_counts.most_common():
        st.sidebar.text(f"{ct}: {count}")

    st.sidebar.markdown("### Synthetic Samples")
    synthetic_count = sum(synthetic_flags)
    st.sidebar.metric("Total Synthetic", synthetic_count)
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

        filtered_samples = []
        for sample in samples:
            if "all" not in filter_content_type:
                if sample.input_content_type not in filter_content_type:
                    continue

            if "all" not in filter_group:
                if sample.meta.get("group") not in filter_group:
                    continue

            if filter_synthetic == "only_synthetic" and not sample.is_synthetic:
                continue
            if filter_synthetic == "only_real" and sample.is_synthetic:
                continue

            filtered_samples.append(sample)

        st.markdown(f"**Showing {len(filtered_samples)} samples**")

        with col2:
            if filtered_samples:
                sample_idx = st.selectbox(
                    "Select Sample",
                    options=range(len(filtered_samples)),
                    format_func=lambda x: f"Conversion ID {filtered_samples[x].id}",
                )

                sample = filtered_samples[sample_idx]

                st.markdown("### Sample Details")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Content Type", sample.input_content_type)
                    st.metric("Group", sample.meta.get("group", "N/A"))
                with col_b:
                    st.metric("Schema ID", sample.schema_id)
                    st.metric("Synthetic", "Yes" if sample.is_synthetic else "No")

                if sample.is_synthetic:
                    st.info(
                        f"Synthetic Type: {sample.meta.get('synthetic_type', 'unknown')}"
                    )

                st.markdown("### Input Data")
                st.code(sample.input_content, language="text")

                st.markdown("### Schema")
                st.json(sample.schema_content)

                st.markdown("### Output")
                try:
                    output = json.loads(sample.output_content)
                    st.json(output)
                except:
                    st.code(sample.output_content, language="text")

                with st.expander("📋 Full Metadata"):
                    st.json(sample.meta)
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
        real_count = len(samples) - synthetic_count
        st.bar_chart({"Real": real_count, "Synthetic": synthetic_count})

    with tab3:
        if st.button("🎲 Get Random Sample"):
            import random

            random_idx = random.randint(0, len(samples) - 1)
            sample = samples[random_idx]

            st.markdown(f"### Conversion ID {sample.id}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Content Type", sample.input_content_type)
                st.metric("Group", sample.meta.get("group", "N/A"))
            with col2:
                st.metric("Schema ID", sample.schema_id)
                st.metric("Synthetic", "Yes" if sample.is_synthetic else "No")

            st.markdown("### Input Data")
            st.code(sample.input_content, language="text")

            st.markdown("### Schema")
            st.json(sample.schema_content)

            st.markdown("### Output")
            try:
                output = json.loads(sample.output_content)
                st.json(output)
            except:
                st.code(sample.output_content, language="text")

            with st.expander("📋 Full Metadata"):
                st.json(sample.meta)


if __name__ == "__main__":
    load_dotenv(override=False)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "INFO"),
    )
    main()
