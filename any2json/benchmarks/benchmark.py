from datetime import datetime
import json
import os
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
from any2json.training.utils import load_hf_dataset
from any2json.utils import configure_loggers, json_dump_safe, logger


model_types = {
    "qwen": QwenVLLMServer,
    "gemini": GeminiModel,
    "vllm_custom": VLLMServerModel,
}


def run_benchmark(model, samples: list[dict]) -> list[dict]:
    results: list[dict] = []

    preds = model.get_predictions(samples)
    id_to_pred = {p["id"]: p for p in preds}
    for i, sample in enumerate(samples):
        prediction = id_to_pred[i]
        input_data = sample["input_data"]
        if isinstance(input_data, dict):
            input_data = json.dumps(input_data)
        results.append(
            {
                "id": i,
                "input_data": input_data,
                "schema": sample["schema"],
                "correct_answer": sample["output"],
                "completion": prediction.get("completion"),
                "answer": prediction.get("answer"),
                "meta": prediction.get("meta"),
                "error": prediction.get("error"),
            }
        )
    return results


def postprocess_answer(answer: str) -> dict | str | list | int | float | bool | None:
    if answer.startswith("```json"):
        answer = answer[7:]
    if answer.endswith("```"):
        answer = answer[:-3]
    return json.loads(answer)


def calculate_metrics(results: list[dict]) -> tuple[list[dict], dict]:
    details_list = []

    if not results:
        return details_list, {
            "percentage_json_errors": 0,
            "percentage_correct": 0,
            "percentage_schema_errors": 0,
            "percentage_request_errors": 0,
            "mean_inference_ms": 0,
        }

    correct = []
    request_error = []
    json_error = []
    schema_error = []
    all_inference_ms = []
    for i, result in enumerate(results):
        details = {}

        if result.get("error"):
            request_error.append(i)
            details["error_type"] = "request_error"
            details["error"] = result["error"]
            details["traceback"] = result["error"]["traceback"]
            details_list.append(details)
            continue

        if isinstance(result["schema"], str):
            result["schema"] = json.loads(result["schema"])
        schema = fastjsonschema.compile(result["schema"])

        try:
            answer = postprocess_answer(result["answer"])
            schema(answer)
        except fastjsonschema.JsonSchemaException as e:
            schema_error.append(i)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
            details["error_type"] = "schema_error"
            details["error"] = str(e)
            details["traceback"] = traceback_str
            details_list.append(details)
            continue
        except Exception as e:
            json_error.append(i)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
            details["error_type"] = "json_error"
            details["error"] = str(e)
            details["traceback"] = traceback_str
            details_list.append(details)
            continue

        correct_answer = result["correct_answer"]

        if isinstance(correct_answer, str):
            correct_answer = json.loads(correct_answer)

        if answer == correct_answer:
            correct.append(i)
            details["correct"] = True
        else:
            details["correct"] = False

        inference_ms = result.get("meta", {}).get("inference_ms")
        if inference_ms:
            all_inference_ms.append(inference_ms)

        details_list.append(details)

    return details_list, {
        "percentage_request_errors": round(len(request_error) / len(results), 3),
        "percentage_json_errors": round(len(json_error) / len(results), 3),
        "percentage_correct": round(len(correct) / len(results), 3),
        "percentage_schema_errors": round(len(schema_error) / len(results), 3),
        "mean_inference_ms": round(np.mean(all_inference_ms).item(), 3),
    }


@click.group()
def cli():
    pass


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
def run(hf_dataset, split, model_type, model_kwargs, output_dir, limit):
    command = " ".join(sys.argv)
    model_kwargs = json.loads(model_kwargs) if model_kwargs else {}

    logger.info(
        f"Benchmarking with inputs: {hf_dataset=}, {split=}, {model_type=}, {model_kwargs=}, {limit=}"
    )
    model_type = model_types[model_type]
    model = model_type(**model_kwargs)

    logger.info(f"Model state: {model.get_state()}")

    samples = load_hf_dataset(hf_dataset)[split].to_list()

    if limit:
        samples = samples[:limit]

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

    details_list, metrics = calculate_metrics(results)

    logger.info(f"Metrics:\n{metrics}")

    output = {
        "metrics": metrics,
        "details": details_list,
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
