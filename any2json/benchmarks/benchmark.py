from datetime import datetime
import json
import os
import logging
import time
import click
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from any2json.benchmarks.models.gemini import GeminiModel
from any2json.benchmarks.models.qwen import QwenModel
from tqdm.auto import tqdm
import fastjsonschema

from any2json.benchmarks.models.vllm_custom import VLLMServerModel
from any2json.utils import configure_loggers, logger


model_types = {
    "qwen": QwenModel,
    "gemini": GeminiModel,
    "vllm_custom": VLLMServerModel,
}


def run_benchmark(model, samples: list[dict]) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    errors: list[dict] = []

    preds, errs = model.get_predictions(samples)
    id_to_pred = {p["id"]: p for p in preds}
    for i, s in enumerate(samples):
        p = id_to_pred.get(i)
        if p:
            input_data = s["input_data"]
            if isinstance(input_data, dict):
                input_data = json.dumps(input_data)
            results.append(
                {
                    "id": i,
                    "input_data": input_data,
                    "schema": s["schema"],
                    "correct_answer": s["output"],
                    "answer": p["answer"],
                    "meta": p.get("meta"),
                }
            )
    for e in errs:
        i = e.get("id")
        s = samples[i] if i is not None and i < len(samples) else {}
        input_data = s.get("input_data") if s else None
        if isinstance(input_data, dict):
            input_data = json.dumps(input_data)
        errors.append(
            {
                "id": i,
                "input_data": input_data,
                "schema": s.get("schema") if s else None,
                "correct_answer": s.get("output") if s else None,
                "error": e.get("error"),
                "traceback": e.get("traceback"),
            }
        )
    return results, errors


def postprocess_answer(answer: str) -> dict | str | list | int | float | bool | None:
    if answer.startswith("```json"):
        answer = answer[7:]
    if answer.endswith("```"):
        answer = answer[:-3]
    return json.loads(answer)


def calculate_metrics(results: list[dict]) -> tuple[list[dict], dict]:
    error = []
    correct = []
    schema_error = []
    for i, result in enumerate(results):
        results[i]["metrics"] = {}
        if isinstance(result["schema"], str):
            result["schema"] = json.loads(result["schema"])

        schema = fastjsonschema.compile(result["schema"])
        try:
            answer = postprocess_answer(result["answer"])
            schema(answer)
        except json.JSONDecodeError as e:
            error.append(i)
            logger.error(e, exc_info=True)
            results[i]["metrics"]["error_type"] = "json_decode_error"
            results[i]["metrics"]["error_message"] = str(e)
            continue
        except fastjsonschema.JsonSchemaException as e:
            schema_error.append(i)
            logger.error(e, exc_info=True)
            results[i]["metrics"]["error_type"] = "schema_error"
            results[i]["metrics"]["error_message"] = str(e)
            continue

        correct_answer = result["correct_answer"]

        if isinstance(correct_answer, str):
            correct_answer = json.loads(correct_answer)

        if answer == correct_answer:
            correct.append(i)
            results[i]["metrics"]["correct"] = True
        else:
            results[i]["metrics"]["correct"] = False

    if len(results) == 0:
        return results, {
            "percentage_json_errors": 0,
            "percentage_correct": 0,
            "percentage_schema_errors": 0,
        }

    return results, {
        "percentage_json_errors": round(len(error) / len(results), 3),
        "percentage_correct": round(len(correct) / len(results), 3),
        "percentage_schema_errors": round(len(schema_error) / len(results), 3),
    }


@click.command()
@click.option(
    "--hf-dataset-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=False,
    help="HF dataset directory to benchmark",
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
def run(hf_dataset_dir, hf_dataset, split, model_type, model_kwargs, output_dir, limit):
    model_kwargs = json.loads(model_kwargs) if model_kwargs else {}

    logger.info(
        f"Benchmarking with inputs: {hf_dataset_dir=}, {hf_dataset=}, {split=}, {model_type=}, {model_kwargs=}, {limit=}"
    )
    model_type = model_types[model_type]
    model = model_type(**model_kwargs)

    logger.info(f"Model state: {model.get_state()}")

    if hf_dataset_dir:
        dataset_dict = DatasetDict.load_from_disk(hf_dataset_dir)
        samples = dataset_dict[split].to_list()
    elif hf_dataset:
        dataset = load_dataset(hf_dataset, split=split)
        samples = dataset.to_list()
    else:
        raise ValueError("Either hf_dataset_dir or hf_dataset must be provided")

    if limit:
        samples = samples[:limit]

    logger.info(f"Running benchmark with {len(samples)} samples")

    start_dt = datetime.now()
    results, errors = run_benchmark(model, samples)
    end_dt = datetime.now()
    logger.info(f"Benchmarking took {end_dt - start_dt}")

    logger.info(f"Obtained {len(results)} results, {len(errors)} errors")
    results, metrics = calculate_metrics(results)

    logger.info(f"Metrics: {metrics}")

    os.makedirs(output_dir, exist_ok=True)

    config = {
        "hf_dataset_dir": hf_dataset_dir,
        "split": split,
        "model_state": model.get_state(),
        "limit": limit,
        "actual_samples": len(samples),
        "start_dt": start_dt.isoformat(),
        "end_dt": end_dt.isoformat(),
        "duration_s": (end_dt - start_dt).total_seconds(),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, "errors.json"), "w") as f:
        json.dump(errors, f, indent=2)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    load_dotenv(override=False)
    configure_loggers(
        level=os.getenv("LOG_LEVEL", "INFO"),
        basic_level=os.getenv("LOG_LEVEL_BASIC", "WARNING"),
    )
    run()
