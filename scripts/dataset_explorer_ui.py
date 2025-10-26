import streamlit as st
import json
from sqlalchemy import select, func, distinct
from any2json.database.client import get_db_session
from any2json.database.models import Chunk, JsonSchema, SchemaConversion, SourceDocument
from any2json.enums import ContentType
import pandas as pd


DB_FILE = "data/database.db"


def init_session():
    if "db_session" not in st.session_state:
        st.session_state.db_session = get_db_session(f"sqlite:///{DB_FILE}")
    return st.session_state.db_session


def get_overview_stats(db_session) -> dict:
    total_chunks = db_session.query(func.count(Chunk.id)).scalar()
    total_schemas = db_session.query(func.count(JsonSchema.id)).scalar()
    total_conversions = db_session.query(func.count(SchemaConversion.id)).scalar()
    total_documents = db_session.query(func.count(SourceDocument.id)).scalar()

    chunks_with_schema = (
        db_session.query(func.count(Chunk.id))
        .filter(Chunk.schema_id.isnot(None))
        .scalar()
    )
    synthetic_chunks = (
        db_session.query(func.count(Chunk.id))
        .filter(Chunk.is_synthetic == True)
        .scalar()
    )
    synthetic_schemas = (
        db_session.query(func.count(JsonSchema.id))
        .filter(JsonSchema.is_synthetic == True)
        .scalar()
    )

    conversions_with_groups = (
        db_session.query(func.count(SchemaConversion.id))
        .filter(SchemaConversion.meta["group"].isnot(None))
        .scalar()
    )

    return {
        "total_chunks": total_chunks,
        "total_schemas": total_schemas,
        "total_conversions": total_conversions,
        "total_documents": total_documents,
        "chunks_with_schema": chunks_with_schema,
        "synthetic_chunks": synthetic_chunks,
        "synthetic_schemas": synthetic_schemas,
        "conversions_with_groups": conversions_with_groups,
    }


def get_content_type_distribution(db_session) -> pd.DataFrame:
    results = (
        db_session.query(Chunk.content_type, func.count(Chunk.id).label("count"))
        .group_by(Chunk.content_type)
        .all()
    )

    return pd.DataFrame(results, columns=["content_type", "count"])


def get_group_distribution(db_session) -> pd.DataFrame:
    query = (
        select(
            func.json_extract(SchemaConversion.meta, "$.group").label("group"),
            func.count().label("count"),
        )
        .where(SchemaConversion.meta["group"].isnot(None))
        .group_by("group")
    )

    results = db_session.execute(query).all()
    return pd.DataFrame(results, columns=["group", "count"])


def browse_chunks(db_session, content_type: str | None, limit: int, offset: int):
    query = select(Chunk).order_by(Chunk.id.desc())

    if content_type and content_type != "All":
        query = query.where(Chunk.content_type == content_type)

    query = query.limit(limit).offset(offset)
    chunks = db_session.execute(query).scalars().all()

    return chunks


def browse_schemas(db_session, limit: int, offset: int):
    query = (
        select(JsonSchema).order_by(JsonSchema.id.desc()).limit(limit).offset(offset)
    )
    schemas = db_session.execute(query).scalars().all()
    return schemas


def browse_conversions(db_session, limit: int, offset: int, with_group_only: bool):
    query = select(SchemaConversion).order_by(SchemaConversion.id.desc())

    if with_group_only:
        query = query.where(SchemaConversion.meta["group"].isnot(None))

    query = query.limit(limit).offset(offset)
    conversions = db_session.execute(query).scalars().all()

    return conversions


def get_export_preview(db_session, num_samples: int):
    query = (
        select(SchemaConversion)
        .where(SchemaConversion.schema_id.isnot(None))
        .where(SchemaConversion.input_chunk_id.isnot(None))
        .where(SchemaConversion.output_chunk_id.isnot(None))
        .where(SchemaConversion.meta["group"].isnot(None))
        .where(
            SchemaConversion.output_chunk.has(
                Chunk.content_type == ContentType.JSON.value
            )
        )
        .order_by(func.random())
        .limit(num_samples)
    )

    conversions = db_session.execute(query).scalars().all()

    samples = []
    for conv in conversions:
        if not conv.input_chunk or not conv.output_chunk or not conv.schema:
            continue

        samples.append(
            {
                "input_data": conv.input_chunk.content,
                "schema": conv.schema.content,
                "output": json.loads(conv.output_chunk.content),
                "meta": {
                    "input_chunk_id": conv.input_chunk.id,
                    "input_chunk_content_type": conv.input_chunk.content_type,
                    "output_chunk_id": conv.output_chunk.id,
                    "schema_id": conv.schema.id,
                    "schema_conversion_id": conv.id,
                    "group": conv.meta.get("group"),
                },
                "input_chunk_meta": conv.input_chunk.meta,
                "output_chunk_meta": conv.output_chunk.meta,
                "schema_meta": conv.schema.meta,
            }
        )

    return samples


def render_overview_page(db_session):
    st.header("📊 Dataset Overview")

    stats = get_overview_stats(db_session)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", f"{stats['total_documents']:,}")
        st.metric("Total Chunks", f"{stats['total_chunks']:,}")
    with col2:
        st.metric("Total Schemas", f"{stats['total_schemas']:,}")
        st.metric("Chunks w/ Schema", f"{stats['chunks_with_schema']:,}")
    with col3:
        st.metric("Total Conversions", f"{stats['total_conversions']:,}")
        st.metric("Conversions w/ Groups", f"{stats['conversions_with_groups']:,}")
    with col4:
        st.metric("Synthetic Chunks", f"{stats['synthetic_chunks']:,}")
        st.metric("Synthetic Schemas", f"{stats['synthetic_schemas']:,}")

    st.subheader("Content Type Distribution")
    content_dist = get_content_type_distribution(db_session)
    st.bar_chart(content_dist.set_index("content_type"))

    st.subheader("Group Distribution")
    group_dist = get_group_distribution(db_session)
    if not group_dist.empty:
        st.bar_chart(group_dist.set_index("group"))
    else:
        st.info("No groups assigned yet")


def render_chunks_page(db_session):
    st.header("📦 Browse Chunks")

    content_types = ["All"] + [ct.value for ct in ContentType]
    selected_type = st.selectbox("Filter by Content Type", content_types)

    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input("Items per page", min_value=1, max_value=100, value=10)
    with col2:
        page = st.number_input("Page", min_value=1, value=1)

    offset = (page - 1) * limit
    chunks = browse_chunks(db_session, selected_type, limit, offset)

    for chunk in chunks:
        with st.expander(f"Chunk {chunk.id} - {chunk.content_type}"):
            st.text(f"ID: {chunk.id}")
            st.text(f"Type: {chunk.content_type}")
            st.text(f"Synthetic: {chunk.is_synthetic}")
            st.text(f"Schema ID: {chunk.schema_id}")
            st.text(f"Parent Document ID: {chunk.parent_document_id}")
            st.text(f"Parent Chunk ID: {chunk.parent_chunk_id}")
            st.code(chunk.content[:1000], language="text")
            if chunk.meta:
                st.json(chunk.meta)


def render_schemas_page(db_session):
    st.header("📋 Browse Schemas")

    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input("Items per page", min_value=1, max_value=100, value=10)
    with col2:
        page = st.number_input("Page", min_value=1, value=1)

    offset = (page - 1) * limit
    schemas = browse_schemas(db_session, limit, offset)

    for schema in schemas:
        num_chunks = len(schema.chunks) if schema.chunks else 0
        with st.expander(f"Schema {schema.id} - {num_chunks} chunks"):
            st.text(f"ID: {schema.id}")
            st.text(f"Synthetic: {schema.is_synthetic}")
            st.text(f"Parent Schema ID: {schema.parent_schema_id}")
            st.text(f"Chunks: {num_chunks}")
            st.json(schema.content)
            if schema.meta:
                st.json(schema.meta)


def render_conversions_page(db_session):
    st.header("🔄 Browse Schema Conversions")

    with_group_only = st.checkbox("Show only conversions with groups", value=True)

    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input("Items per page", min_value=1, max_value=100, value=10)
    with col2:
        page = st.number_input("Page", min_value=1, value=1)

    offset = (page - 1) * limit
    conversions = browse_conversions(db_session, limit, offset, with_group_only)

    for conv in conversions:
        group = conv.meta.get("group") if conv.meta else None
        with st.expander(f"Conversion {conv.id} - Group: {group}"):
            st.text(f"ID: {conv.id}")
            st.text(f"Group: {group}")

            if conv.input_chunk:
                st.subheader("Input Chunk")
                st.text(f"Type: {conv.input_chunk.content_type}")
                st.code(conv.input_chunk.content[:500], language="text")

            if conv.schema:
                st.subheader("Schema")
                st.json(conv.schema.content)

            if conv.output_chunk:
                st.subheader("Output Chunk")
                st.text(f"Type: {conv.output_chunk.content_type}")
                st.code(conv.output_chunk.content[:500], language="text")

            if conv.meta:
                st.subheader("Metadata")
                st.json(conv.meta)


def render_export_preview_page(db_session):
    st.header("🚀 Export Preview")

    st.write("Preview what samples will look like when exported to HF format")

    num_samples = st.number_input(
        "Number of samples to preview", min_value=1, max_value=50, value=5
    )

    if st.button("Generate Preview"):
        samples = get_export_preview(db_session, num_samples)

        st.subheader(f"Preview of {len(samples)} samples")

        for i, sample in enumerate(samples):

            sample_id = sample["meta"]["schema_conversion_id"]
            group = sample["meta"].get("group")
            input_content_type = sample["meta"]["input_chunk_content_type"]

            with st.expander(
                f"Sample {sample_id} - Group: {group} - Content Type: {input_content_type}"
            ):
                st.subheader("Input Data")
                st.code(sample["input_data"], language="text")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Schema")
                    st.json(sample["schema"])

                with col2:
                    st.subheader("Output")
                    st.json(sample["output"])

                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Sample metadata")
                    st.json(sample["meta"])
                    st.subheader("Input Chunk Metadata")
                    st.json(sample["input_chunk_meta"])
                with col2:
                    st.subheader("Output Chunk Metadata")
                    st.json(sample["output_chunk_meta"])
                    st.subheader("Schema Metadata")
                    st.json(sample["schema_meta"])


def main():
    st.set_page_config(
        page_title="Any2JSON Dataset Explorer",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Any2JSON Dataset Explorer")

    db_session = init_session()

    page = st.sidebar.radio(
        "Navigation", ["Overview", "Chunks", "Schemas", "Conversions", "Export Preview"]
    )

    if page == "Overview":
        render_overview_page(db_session)
    elif page == "Chunks":
        render_chunks_page(db_session)
    elif page == "Schemas":
        render_schemas_page(db_session)
    elif page == "Conversions":
        render_conversions_page(db_session)
    elif page == "Export Preview":
        render_export_preview_page(db_session)


if __name__ == "__main__":
    main()
