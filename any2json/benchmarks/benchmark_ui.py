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
        predictions = benchmark_results["results"]
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


def to_predictions_df(results_per_model: dict[str, dict]):
    predictions_records = []
    for model_name, benchmark_results in results_per_model.items():
        predictions = benchmark_results["results"]
        metrics_details = benchmark_results["metrics_details"]
        for prediction in predictions:
            sample_id = prediction["id"]

            try:
                metric_metails = metrics_details[sample_id]
            except IndexError:
                metric_metails = None

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
    df["correct_answer"] = df["correct_answer"].apply(lambda x: json.dumps(x, indent=1))
    df["schema"] = df["schema"].apply(
        lambda x: json.dumps(x, indent=1) if x and x != SCHEMA_MISSING_TOKEN else x
    )
    df["answer"] = df["answer"].apply(lambda x: json.dumps(x, indent=1))
    df["meta"] = df["meta"].apply(lambda x: json.dumps(x, indent=1))
    df["metrics_details"] = df["metrics_details"].apply(
        lambda x: json.dumps(x, indent=1)
    )

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

    # predictions_records = []
    # for model_name, benchmark_results in results_per_model.items():
    #     predictions = benchmark_results["results"]
    #     metrics_details = benchmark_results["metrics_details"]
    #     for prediction in predictions:
    #         sample_id = prediction["id"]
    #         predictions_records.append(
    #             {
    #                 "model_name": model_name,
    #                 "sample_id": sample_id,
    #                 "input_data": prediction["input_data"],
    #                 "schema": prediction["schema"],
    #                 "answer": prediction["answer"],
    #                 "correct_answer": prediction["correct_answer"],
    #                 "meta": prediction["meta"],
    #                 "metrics_details": metrics_details[sample_id],
    #             }
    #         )

    # df = pd.DataFrame(predictions_records)
    df = to_predictions_df(results_per_model)
    df["correct_answer"] = df["correct_answer"].apply(lambda x: json.dumps(x, indent=1))
    df["schema"] = df["schema"].apply(
        lambda x: json.dumps(x, indent=1) if x and x != SCHEMA_MISSING_TOKEN else x
    )
    df["answer"] = df["answer"].apply(lambda x: json.dumps(x, indent=1))
    df["meta"] = df["meta"].apply(lambda x: json.dumps(x, indent=1))
    df["metrics_details"] = df["metrics_details"].apply(
        lambda x: json.dumps(x, indent=1)
    )

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
            st.text(first_row["input_data"])

    with col2:
        st.markdown("#### Schema")
        with st.expander("View Schema", expanded=False):
            try:
                schema_obj = first_row["schema"]
                st.json(schema_obj)
            except json.JSONDecodeError:
                st.text(first_row["schema"])

    with col3:
        st.markdown("#### Correct Answer")
        with st.expander("View Correct Answer", expanded=False):
            try:
                correct_answer_obj = json.loads(first_row["correct_answer"])
                st.json(correct_answer_obj)
            except json.JSONDecodeError:
                st.text(first_row["correct_answer"])

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

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model Answer:**")
            try:
                answer_obj = json.loads(model_row["answer"])
                st.json(answer_obj)
            except json.JSONDecodeError:
                st.text(model_row["answer"])

            st.markdown("**Metrics Details:**")
            st.json(model_row["metrics_details"])

        with col2:
            st.markdown("**Diff with Correct Answer:**")
            try:
                correct_formatted = json.dumps(
                    json.loads(first_row["correct_answer"]), indent=2, sort_keys=True
                )
            except json.JSONDecodeError:
                correct_formatted = first_row["correct_answer"]

            try:
                answer_formatted = json.dumps(
                    json.loads(model_row["answer"]), indent=2, sort_keys=True
                )
            except json.JSONDecodeError:
                answer_formatted = model_row["answer"]

            diff = list(
                difflib.unified_diff(
                    correct_formatted.splitlines(keepends=True),
                    answer_formatted.splitlines(keepends=True),
                    fromfile="Correct Answer",
                    tofile=f"{model_name} Answer",
                    lineterm="",
                )
            )

            if diff:
                diff_text = "".join(diff)
                st.code(diff_text, language="diff")
            else:
                st.success("✅ Exact match with correct answer!")

        st.divider()


def show_error_analysis(results_per_model: dict[str, dict]):
    st.markdown("### Error Analysis")

    st.markdown(f"#### Input data len vs error rates")

    plot_df = to_predictions_df(results_per_model)

    plot_df["error_type"] = plot_df.metrics_details.apply(
        lambda x: x.get("error_type")
        or ("wrong_content" if x.get("correct") is False else None)
    )
    plot_df = plot_df[plot_df.error_type.notna()]
    plot_df["input_data_len"] = plot_df["input_data"].apply(len)

    plot_df["input_data_len_bucket"] = pd.cut(
        plot_df["input_data_len"],
        bins=range(0, plot_df["input_data_len"].max() + 1, 500),
    )
    plot_df = (
        plot_df.groupby(["model_name", "input_data_len_bucket", "error_type"])
        .size()
        .reset_index(name="count")
    )
    plot_df["input_data_len_bucket"] = plot_df["input_data_len_bucket"].astype(str)
    plot_df["model_name"] = plot_df["model_name"].astype(str)
    fig = px.bar(
        plot_df,
        x="input_data_len_bucket",
        y="count",
        color="error_type",
        barmode="stack",
        facet_col="model_name",
        width=600,
        height=400,
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
