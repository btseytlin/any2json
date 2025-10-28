import os
import sys
import click
from dotenv import load_dotenv
import pandas as pd
from any2json.training.constants import SCHEMA_MISSING_TOKEN
from any2json.utils import configure_loggers, json_dump_safe, logger
from any2json.benchmarks.benchmark import calculate_metrics
import json
import streamlit as st
from plotly import express as px
import difflib

DEFAULT_JSON_EXPANDED = 2
DEFAULT_JSON_INDENT = 1


def render_json_or_text(json_obj: dict | str):
    if isinstance(json_obj, str):
        try:
            json.loads(json_obj)
            return st.json(json_obj, expanded=DEFAULT_JSON_EXPANDED)
        except json.JSONDecodeError:
            return st.text(json_obj)
    return st.json(json_obj, expanded=DEFAULT_JSON_EXPANDED)


def process_error_type(metrics_details: dict):
    error_type = metrics_details.get("error_type")
    correct = metrics_details.get("correct")
    if error_type:
        return error_type

    if not error_type and correct is False:
        return "wrong_content"
    return "None"


def load_results_per_model(results_dir: str) -> dict[str, dict]:
    results_per_model = {}
    for subdir in os.scandir(results_dir):
        if not subdir.is_dir():
            continue
        info = json.load(open(os.path.join(subdir, "info.json")))
        results = json.load(open(os.path.join(subdir, "results.json")))
        results_per_model[subdir.name] = {
            "info": info,
            "results": results,
        }
    return results_per_model


def show_metrics_table(results_per_model: dict[str, dict]):
    benchmark_durations = {}
    metric_records = []
    for model_name, benchmark_results in results_per_model.items():
        info = benchmark_results["info"]
        benchmark_durations[model_name] = info["duration_s"]
        # predictions = benchmark_results["results"]
        metrics = benchmark_results["metrics"]
        for metric_name, metric_value in metrics.items():
            metric_records.append(
                {
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                }
            )

    df = pd.DataFrame(metric_records)

    df = df.pivot(index="model_name", columns="metric_name", values="metric_value")

    st.markdown("### Metrics")

    st.dataframe(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Latency vs quality")
        fig = px.scatter(
            df.reset_index(),
            x="inference_ms_mean",
            y="percentage_correct",
            color="model_name",
            width=600,
            height=400,
        )
        st.plotly_chart(fig)

    with col2:
        st.markdown("### Error type distribution")

        error_cols = [
            "percentage_json_errors",
            "percentage_schema_errors",
            "percentage_request_errors",
            "percentage_content_errors",
        ]

        # Stacked bar chart of error distributions for each model
        plot_df = pd.DataFrame(metric_records)
        plot_df = plot_df[plot_df["metric_name"].isin(error_cols)]
        fig = px.bar(
            plot_df,
            x="model_name",
            y="metric_value",
            color="metric_name",
            width=600,
            height=400,
        )
        st.plotly_chart(fig)


def show_info(results_per_model: dict[str, dict]):
    st.markdown("### Info")
    with st.expander("View info", expanded=False):
        for model_name, benchmark_results in results_per_model.items():
            info = benchmark_results["info"]
            st.markdown(f"### {model_name}")
            st.json(info)


def predictions_df_to_presentation_df(df: pd.DataFrame):
    df = df.copy()
    df["correct_answer"] = df["correct_answer"].apply(
        lambda x: json.dumps(x, indent=DEFAULT_JSON_INDENT, sort_keys=True)
    )
    df["schema"] = df["schema"].apply(
        lambda x: (
            json.dumps(x, indent=DEFAULT_JSON_INDENT, sort_keys=True)
            if x and x != SCHEMA_MISSING_TOKEN
            else x
        )
    )
    df["answer"] = df["answer"].apply(
        lambda x: json.dumps(x, indent=DEFAULT_JSON_INDENT, sort_keys=True)
    )
    df["meta"] = df["meta"].apply(
        lambda x: json.dumps(x, indent=DEFAULT_JSON_INDENT, sort_keys=True)
    )
    df["metrics_details"] = df["metrics_details"].apply(
        lambda x: json.dumps(x, indent=DEFAULT_JSON_INDENT, sort_keys=True)
    )
    return df


def to_predictions_df(results_per_model: dict[str, dict]):
    predictions_records = []
    for model_name, benchmark_results in results_per_model.items():
        predictions = benchmark_results["results"]
        metrics_details = benchmark_results["metrics_details"]
        for prediction in predictions:
            sample_id = prediction["sample_id"]

            metric_metails = metrics_details[sample_id]

            predictions_records.append(
                {
                    "model_name": model_name,
                    "sample_id": sample_id,
                    "input_data": prediction["input_data"],
                    "schema": prediction["schema"] or SCHEMA_MISSING_TOKEN,
                    "answer": prediction["answer"],
                    "correct_answer": prediction["correct_answer"],
                    "meta": prediction["meta"] or {},
                    "metrics_details": metric_metails,
                }
            )

    df = pd.DataFrame(predictions_records)
    df = df.fillna("")

    return df


def show_predictions(results_per_model: dict[str, dict]):
    df = to_predictions_df(results_per_model)
    df = predictions_df_to_presentation_df(df)
    st.markdown("### Predictions")
    st.dataframe(
        df,
        column_config={
            "correct_answer": st.column_config.JsonColumn(),
            "schema": st.column_config.JsonColumn(),
            "answer": st.column_config.JsonColumn(),
            "meta": st.column_config.JsonColumn(),
            "metrics_details": st.column_config.JsonColumn(),
        },
    )


def show_prediction_explorer(results_per_model: dict[str, dict]):
    st.markdown("### Prediction Explorer")

    only_errors = st.checkbox("Only errors", value=True)

    df = to_predictions_df(results_per_model)

    if only_errors:
        df = df[df["metrics_details"].apply(lambda x: x.get("correct") is not True)]

    available_sample_ids = sorted(df["sample_id"].unique())
    available_models = sorted(df["model_name"].unique())

    col1, col2 = st.columns(2)

    with col1:
        selected_sample_id = st.selectbox(
            "Select Sample ID:", options=available_sample_ids, index=0
        )

    with col2:
        selected_models = st.multiselect(
            "Select Models:",
            options=available_models,
            default=(
                available_models[:2] if len(available_models) > 1 else available_models
            ),
        )

    if not selected_models:
        st.warning("Please select at least one model.")
        return

    sample_data = df[df["sample_id"] == selected_sample_id]

    if sample_data.empty:
        st.error(f"No data found for sample ID {selected_sample_id}")
        return

    first_row = sample_data.iloc[0]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Input Data")
        with st.expander("View Input Data", expanded=False):
            render_json_or_text(first_row["input_data"])

    with col2:
        st.markdown("#### Schema")
        with st.expander("View Schema", expanded=False):
            render_json_or_text(first_row["schema"])

    with col3:
        st.markdown("#### Meta")
        with st.expander("Metrics Details", expanded=False):
            render_json_or_text(first_row["metrics_details"])

    st.markdown("#### Model Predictions and Diffs")

    for model_name in selected_models:
        model_data = sample_data[sample_data["model_name"] == model_name]

        if model_data.empty:
            st.warning(
                f"No data found for model {model_name} on sample {selected_sample_id}"
            )
            continue

        model_row = model_data.iloc[0]

        st.markdown(f"##### Model: {model_name}")

        col1, col2, col3 = st.columns(3)

        correct_answer = json.dumps(
            model_row["correct_answer"],
            indent=DEFAULT_JSON_INDENT,
            sort_keys=True,
        )

        try:
            answer = json.dumps(
                json.loads(model_row["answer"]),
                indent=DEFAULT_JSON_INDENT,
                sort_keys=True,
            )
        except json.JSONDecodeError:
            answer = model_row["answer"]

        with col1:
            st.markdown("**Correct Answer:**")
            render_json_or_text(correct_answer)

        with col2:
            st.markdown("**Model Answer:**")
            render_json_or_text(answer)

        with col3:
            st.markdown("**Diff with Correct Answer:**")
            diff = list(
                difflib.unified_diff(
                    correct_answer.splitlines(keepends=True),
                    answer.splitlines(keepends=True),
                    fromfile="Correct Answer",
                    tofile=f"Answer {model_name}",
                    lineterm="",
                )
            )

            if diff:
                diff_text = "".join(diff)
                st.code(diff_text, language="diff")
            else:
                st.success("✅ Exact match with correct answer!")

        st.divider()


def prepare_error_buckets_df(results_per_model: dict[str, dict]):
    preds_df = to_predictions_df(results_per_model)
    preds_df["error_type"] = preds_df.metrics_details.apply(process_error_type)
    preds_df["input_data_len"] = (
        preds_df["input_data"].astype(str).apply(len).astype(int)
    )
    preds_df["input_data_len_bucket"] = pd.cut(
        preds_df["input_data_len"],
        bins=range(0, preds_df["input_data_len"].max() + 1, 500),
    )

    sample_counts = (
        preds_df.groupby(["model_name", "input_data_len_bucket"])
        .size()
        .reset_index(name="sample_count")
    )

    error_counts = (
        preds_df[preds_df.error_type.notna()]
        .groupby(["model_name", "input_data_len_bucket", "error_type"])
        .size()
        .reset_index(name="error_count")
    )
    error_counts["input_data_len_bucket"] = error_counts[
        "input_data_len_bucket"
    ].astype(str)
    error_counts["model_name"] = error_counts["model_name"].astype(str)

    sample_counts["input_data_len_bucket"] = sample_counts[
        "input_data_len_bucket"
    ].astype(str)
    sample_counts["model_name"] = sample_counts["model_name"].astype(str)

    plot_df = error_counts.merge(
        sample_counts, on=["model_name", "input_data_len_bucket"]
    )
    plot_df["error_rate"] = plot_df["error_count"] / plot_df["sample_count"] * 100
    return plot_df


def show_error_analysis(results_per_model: dict[str, dict]):
    st.markdown("### Error Analysis")

    plot_df = prepare_error_buckets_df(results_per_model)

    if not plot_df["error_type"].any() or plot_df["error_count"].sum() == 0:
        st.warning("No errors found")
        return

    st.markdown(f"#### Input data len vs error rates")

    fig = px.bar(
        plot_df,
        x="input_data_len_bucket",
        y="error_count",
        color="error_type",
        barmode="stack",
        facet_col="model_name",
        width=600,
        height=400,
    )
    st.plotly_chart(fig)

    st.markdown(f"#### Input data len vs error rates (normalized)")

    fig = px.bar(
        plot_df,
        x="input_data_len_bucket",
        y="error_rate",
        color="error_type",
        barmode="stack",
        facet_col="model_name",
        width=600,
        height=400,
        labels={"error_rate": "Error Rate (%)"},
    )
    st.plotly_chart(fig)

    df = to_predictions_df(results_per_model)

    st.markdown(f"#### Input data len vs inference time")

    plot_df = df[df.meta.apply(lambda x: x.get("inference_ms")).notna()]
    plot_df["input_data_len"] = plot_df["input_data"].apply(len)
    plot_df["inference_ms"] = plot_df["meta"].apply(lambda x: x.get("inference_ms"))
    fig = px.scatter(
        plot_df,
        x="input_data_len",
        y="inference_ms",
        color="model_name",
        width=600,
        height=400,
    )
    st.plotly_chart(fig)

    st.markdown(f"#### Percentage Correct by Input Content Type")

    content_type_df = df.copy()
    content_type_df["correct"] = content_type_df["metrics_details"].apply(
        lambda x: x.get("correct", False)
    )
    content_type_df["content_type"] = content_type_df["metrics_details"].apply(
        lambda x: x.get("sample_meta", {}).get("input_chunk_content_type", "unknown")
    )

    content_type_stats = (
        content_type_df.groupby(["model_name", "content_type"])["correct"]
        .agg(["sum", "count"])
        .reset_index()
    )
    content_type_stats["percentage_correct"] = (
        content_type_stats["sum"] / content_type_stats["count"] * 100
    )

    if not content_type_stats.empty:
        fig = px.bar(
            content_type_stats,
            x="content_type",
            y="percentage_correct",
            color="model_name",
            barmode="group",
            width=800,
            height=400,
            labels={"percentage_correct": "Percentage Correct (%)"},
        )
        st.plotly_chart(fig)
    else:
        st.warning("No content type data available")

    st.markdown(f"#### Percentage Correct by Synthetic Type")

    synthetic_type_df = df.copy()
    synthetic_type_df["correct"] = synthetic_type_df["metrics_details"].apply(
        lambda x: x.get("correct", False)
    )
    synthetic_type_df["synthetic_type"] = synthetic_type_df["metrics_details"].apply(
        lambda x: x.get("sample_meta", {}).get("synthetic_type", "unknown")
    )

    synthetic_type_stats = (
        synthetic_type_df.groupby(["model_name", "synthetic_type"])["correct"]
        .agg(["sum", "count"])
        .reset_index()
    )
    synthetic_type_stats["percentage_correct"] = (
        synthetic_type_stats["sum"] / synthetic_type_stats["count"] * 100
    )

    if not synthetic_type_stats.empty:
        fig = px.bar(
            synthetic_type_stats,
            x="synthetic_type",
            y="percentage_correct",
            color="model_name",
            barmode="group",
            width=800,
            height=400,
            labels={"percentage_correct": "Percentage Correct (%)"},
        )
        st.plotly_chart(fig)
    else:
        st.warning("No synthetic type data available")


def visualize_results(results_dir: str, results_per_model: dict[str, dict]):
    st.set_page_config(page_title="Any2JSON Inference", page_icon="🔄", layout="wide")

    st.markdown(f"Results Directory: {results_dir}")

    st.title("🔄 Any2JSON Model Benchmark Results")

    show_metrics_table(results_per_model)

    show_info(results_per_model)

    show_predictions(results_per_model)

    show_prediction_explorer(results_per_model)

    show_error_analysis(results_per_model)


def main():
    results_dir = sys.argv[1]
    print(f"Results directory: {results_dir}")
    results_per_model = load_results_per_model(results_dir)
    for model_name, results in results_per_model.items():
        print(f"Model name: {model_name}")
        info = results["info"]
        predictions = results["results"]
        (
            results_per_model[model_name]["metrics_details"],
            results_per_model[model_name]["metrics"],
        ) = calculate_metrics(predictions)

    visualize_results(results_dir, results_per_model)


if __name__ == "__main__":
    load_dotenv(override=False)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    main()
