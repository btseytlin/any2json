from datetime import datetime
import difflib
import json
import os
import random
import sys
import numpy as np
import traceback
import click
from dotenv import load_dotenv
from any2json.benchmarks.models.gemini import GeminiModel
from any2json.benchmarks.models.qwen import QwenVLLMServer
from tqdm.auto import tqdm
import fastjsonschema

from any2json.benchmarks.models.vllm_custom import VLLMServerModel
from any2json.training.constants import SCHEMA_MISSING_TOKEN
from any2json.training.utils import load_hf_dataset
from any2json.utils import (
    configure_loggers,
    json_dumps_minified,
    json_dump_safe,
    logger,
)


model_types = {
    "qwen": QwenVLLMServer,
    "gemini": GeminiModel,
    "vllm_custom": VLLMServerModel,
}


def run_benchmark(model: VLLMServerModel, samples: list[dict]) -> list[dict]:
    results: list[dict] = []
    preds = model.get_predictions(samples)
    id_to_pred = {p["sample_id"]: p for p in preds}
    for sample in samples:
        sample_id = sample["sample_id"]
        try:
            prediction = id_to_pred[sample_id]
            input_data = sample["input_data"]
            if isinstance(input_data, dict):
                input_data = json.dumps(input_data)
            results.append(
                {
                    "sample_id": sample_id,
                    "input_data": input_data,
                    "schema": sample["schema"],
                    "correct_answer": sample["output"],
                    "completion": prediction.get("completion"),
                    "answer": prediction.get("answer"),
                    "meta": prediction.get("meta"),
                    "sample_meta": sample.get("meta"),
                    "error": prediction.get("error"),
                }
            )
        except Exception as e:
            logger.error(
                f"Error processing prediction for sample {sample_id}: {e}",
                exc_info=True,
            )
            continue
    return results


def postprocess_answer(answer: str) -> dict | str | list | int | float | bool | None:
    if answer.startswith("```json"):
        answer = answer[7:]
    if answer.endswith("```"):
        answer = answer[:-3]
    return json.loads(answer)


def get_levenstein_distance(true: dict, predicted: dict) -> int:
    true_str = json_dumps_minified(true)
    predicted_str = json_dumps_minified(predicted)
    return round(difflib.SequenceMatcher(None, true_str, predicted_str).ratio(), 4)


def count_different_chars(true: dict, predicted: dict) -> int:
    true_str = json_dumps_minified(true)
    predicted_str = json_dumps_minified(predicted)
    matcher = difflib.SequenceMatcher(None, true_str, predicted_str)
    matches = sum(triple[-1] for triple in matcher.get_matching_blocks())
    return min(max(len(true_str) - matches, 0), len(true_str))


def calculate_diff_metrics(
    answer: dict, correct_answer: dict
) -> dict[str, float | int]:

    levenstein_distance = get_levenstein_distance(correct_answer, answer)
    different_chars_true = count_different_chars(correct_answer, answer)
    different_chars_predicted = count_different_chars(answer, correct_answer)

    correct_answer_dumped = json.dumps(correct_answer, indent=1, sort_keys=True)
    answer_dumped = json.dumps(answer, indent=1, sort_keys=True)

    diff_parts = list(
        difflib.unified_diff(
            correct_answer_dumped.splitlines(keepends=True),
            answer_dumped.splitlines(keepends=True),
            fromfile="Correct Answer",
            tofile="Model Answer",
            lineterm="",
            n=0,
        )
    )

    added_lines = [part for part in diff_parts[3:] if part.startswith("+")]
    removed_lines = [part for part in diff_parts[3:] if part.startswith("-")]

    diff_size_lines_added = len(added_lines)
    diff_size_lines_removed = len(removed_lines)
    diff_size_lines = max(diff_size_lines_added, diff_size_lines_removed)

    return {
        "diff_size_lines": diff_size_lines,
        "diff_size_chars_added": different_chars_predicted,
        "diff_size_chars_missing": different_chars_true,
        "levenstein_distance": levenstein_distance,
    }


def calculate_sample_metrics(result: dict) -> dict:
    metrics_details = {}

    if result.get("error"):
        metrics_details["error_type"] = "request_error"
        metrics_details["error"] = result["error"]
        try:
            metrics_details["traceback"] = result["error"]["traceback"]
        except Exception:
            metrics_details["traceback"] = None
        return metrics_details

    schema_validator = None
    if result["schema"] != SCHEMA_MISSING_TOKEN:
        if isinstance(result["schema"], str):
            result["schema"] = json.loads(result["schema"])
        schema_validator = fastjsonschema.compile(result["schema"])

    try:
        answer = postprocess_answer(result["answer"])
        if schema_validator:
            schema_validator(answer)
    except fastjsonschema.JsonSchemaException as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_str = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        metrics_details["error_type"] = "schema_error"
        metrics_details["error"] = str(e)
        metrics_details["traceback"] = traceback_str
        return metrics_details
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_str = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        metrics_details["error_type"] = "json_error"
        metrics_details["error"] = str(e)
        metrics_details["traceback"] = traceback_str
        return metrics_details

    correct_answer = result["correct_answer"]

    if isinstance(correct_answer, str):
        correct_answer = json.loads(correct_answer)

    metrics_details["correct"] = answer == correct_answer

    diff_metrics = calculate_diff_metrics(answer, correct_answer)
    metrics_details.update(diff_metrics)

    metrics_details["inference_ms"] = result.get("meta", {}).get("inference_ms")

    return metrics_details


def calculate_metrics(results: list[dict]) -> tuple[list[dict], dict]:
    if not results:
        return [], {}

    metrics_details = {}
    for result in results:
        sample_metric_details = calculate_sample_metrics(result)
        metrics_details[result["sample_id"]] = sample_metric_details

    metrics_details_list = list(metrics_details.values())

    aggregate_metrics = {}
    request_error = [
        m for m in metrics_details_list if m.get("error_type") == "request_error"
    ]
    aggregate_metrics["percentage_request_errors"] = round(
        len(request_error) / len(results), 3
    )

    json_error = [
        m for m in metrics_details_list if m.get("error_type") == "json_error"
    ]
    aggregate_metrics["percentage_json_errors"] = round(
        len(json_error) / len(results), 3
    )

    schema_error = [
        m for m in metrics_details_list if m.get("error_type") == "schema_error"
    ]
    aggregate_metrics["percentage_schema_errors"] = round(
        len(schema_error) / len(results), 3
    )

    correct = [m for m in metrics_details_list if m.get("correct")]
    aggregate_metrics["percentage_correct"] = round(len(correct) / len(results), 3)

    statistics_metrics = [
        "diff_size_lines",
        "diff_size_chars_added",
        "diff_size_chars_missing",
        "levenstein_distance",
        "inference_ms",
    ]

    for metric in statistics_metrics:
        all_metrics = [
            m.get(metric) for m in metrics_details_list if m.get(metric) is not None
        ]
        aggregate_metrics[f"{metric}_mean"] = round(np.mean(all_metrics).item(), 3)

    return metrics_details, aggregate_metrics


@click.group()
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed",
)
def cli(seed: int):
    global SEED
    SEED = seed

    random.seed(SEED)

    logger.info(f"Using seed: {SEED}")


@cli.command(
    name="run",
)
@click.option(
    "--hf-dataset",
    default="btseytlin/any2json",
    type=str,
    required=False,
    help="HF dataset to benchmark",
)
@click.option(
    "--split",
    default="test",
    type=str,
    help="Split to benchmark",
)
@click.option(
    "--model-type",
    default="gemini",
    type=str,
    help="Model type to benchmark",
)
@click.option(
    "--model-kwargs",
    default=None,
    type=str,
    help="Model kwargs in JSON format",
)
@click.option(
    "--output-dir",
    default="./benchmark_results",
    type=click.Path(),
    help="Output directory to save the results",
)
@click.option(
    "--limit",
    type=int,
    help="Limit the number of prompts to benchmark",
)
@click.option(
    "--run-id",
    type=str,
    help="WANDB run id",
)
def run(hf_dataset, split, model_type, model_kwargs, output_dir, limit, run_id):
    if run_id:
        import wandb

        wandb.init(id=run_id, resume="allow")
        logger.debug(f"Resumed wandb run: {run_id}")

    command = " ".join(sys.argv)
    model_kwargs = json.loads(model_kwargs) if model_kwargs else {}

    logger.info(
        f"Benchmarking with inputs: {hf_dataset=}, {split=}, {model_type=}, {model_kwargs=}, {limit=}"
    )
    model_type = model_types[model_type]
    model = model_type(**model_kwargs)

    logger.info(f"Model state: {model.get_state()}")

    samples = load_hf_dataset(hf_dataset)[split].to_list()

    sample_indices = list(range(len(samples)))
    if limit:
        sample_indices = random.sample(sample_indices, limit)
        samples = [samples[i] for i in sample_indices]

    for sample_id, sample in zip(sample_indices, samples, strict=True):
        sample["sample_id"] = sample_id
        if (
            isinstance(sample["schema"], str)
            and sample["schema"] != SCHEMA_MISSING_TOKEN
        ):
            sample["schema"] = json.loads(sample["schema"])
        if isinstance(sample["output"], str):
            sample["output"] = json.loads(sample["output"])
        if isinstance(sample["meta"], str):
            sample["meta"] = json.loads(sample["meta"])

    logger.info(f"Running benchmark with {len(samples)} samples")

    start_dt = datetime.now()
    results = run_benchmark(model, samples)
    end_dt = datetime.now()
    duration_s = (end_dt - start_dt).total_seconds()
    logger.info(f"Benchmarking took {duration_s} seconds")

    logger.info(f"Obtained {len(results)} results")

    run_config = {
        "command": command,
        "hf_dataset": hf_dataset,
        "split": split,
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "output_dir": output_dir,
        "limit": limit,
    }

    errors = [r for r in results if r.get("error")]

    run_info = {
        "config": run_config,
        "model_state": model.get_state(),
        "num_samples": len(samples),
        "num_results": len(results),
        "num_errors": len(errors),
        "start_dt": start_dt.isoformat(),
        "end_dt": end_dt.isoformat(),
        "duration_s": duration_s,
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json_dump_safe(
            run_info,
            f,
            indent=2,
        )

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json_dump_safe(
            results,
            f,
            indent=2,
        )


@cli.command(
    name="metrics",
)
@click.argument(
    "results_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
def calculate_metrics_cmd(results_dir):
    with open(os.path.join(results_dir, "results.json"), "r") as f:
        results = json.load(f)

    details, metrics = calculate_metrics(results)

    logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    output = {
        "metrics": metrics,
        "details": details,
    }

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json_dump_safe(output, f, indent=2)


if __name__ == "__main__":
    load_dotenv(override=False)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    cli()
