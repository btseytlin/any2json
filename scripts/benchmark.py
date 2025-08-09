import json
import os
import logging
import sys
import traceback
import click
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from any2json.models.gemini import GeminiModel
from any2json.models.qwen import QwenModel
from tqdm.auto import tqdm
import fastjsonschema

from any2json.utils import configure_loggers, logger


model_types = {
    "qwen": QwenModel,
    "gemini": GeminiModel,
}


def run_benchmark(model, samples: list[dict]) -> list[dict]:
    results = []
    errors = []
    for i, sample in tqdm(
        enumerate(samples),
        total=len(samples),
        desc="Benchmarking",
    ):
        input_data: str | dict = sample["input_data"]

        if isinstance(input_data, dict):
            input_data = json.dumps(input_data)

        schema: dict = sample["schema"]
        correct_answer: dict = sample["output"]

        try:
            answer, meta = model.convert_to_json(input_data, schema)

            results.append(
                {
                    "id": i,
                    "input_data": input_data,
                    "schema": schema,
                    "correct_answer": correct_answer,
                    "answer": answer,
                    "meta": meta,
                }
            )
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            errors.append(
                {
                    "id": i,
                    "input_data": input_data,
                    "schema": schema,
                    "correct_answer": correct_answer,
                    "error": str(e),
                    "traceback": "".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    ),
                }
            )
        except KeyboardInterrupt:
            break

    return results, errors


def calculate_metrics(results):
    error = []
    correct = []
    schema_error = []
    for i, result in enumerate(results):
        if isinstance(result["schema"], str):
            result["schema"] = json.loads(result["schema"])

        schema = fastjsonschema.compile(result["schema"])
        try:
            answer = json.loads(result["answer"])

            schema(answer)

            correct_answer = result["correct_answer"]

            if isinstance(correct_answer, str):
                correct_answer = json.loads(correct_answer)

            if answer == correct_answer:
                correct.append(i)
        except json.JSONDecodeError as e:
            error.append(i)
        except fastjsonschema.JsonSchemaException as e:
            schema_error.append(i)

    return {
        "percentage_errors": len(error) / len(results),
        "percentage_correct": len(correct) / len(results),
        "percentage_schema_errors": len(schema_error) / len(results),
    }


@click.command()
@click.option(
    "--hf-dataset-dir",
    default="data/hf_dataset",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=True,
    help="HF dataset directory to benchmark",
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
    "--model-name",
    default=None,
    type=str,
    help="Model name to benchmark",
)
@click.option(
    "--output-dir",
    default="data/final/benchmark_results",
    type=click.Path(),
    help="Output directory to save the results",
)
@click.option(
    "--limit",
    type=int,
    help="Limit the number of prompts to benchmark",
)
def run(hf_dataset_dir, split, model_type, model_name, output_dir, limit):
    logger.info(
        f"Benchmarking {hf_dataset_dir} split {split} with {model_type} {model_name}"
    )

    model_type = model_types[model_type]
    model = model_type(model_name=model_name)

    dataset_dict = DatasetDict.load_from_disk(hf_dataset_dir)

    samples = dataset_dict[split].to_list()

    if limit:
        samples = samples[:limit]

    results, errors = run_benchmark(model, samples)

    metrics = calculate_metrics(results)

    logger.info(f"Metrics: {metrics}")

    os.makedirs(output_dir, exist_ok=True)

    config = {
        "hf_dataset_dir": hf_dataset_dir,
        "split": split,
        "model_state": model.get_state(),
        "limit": limit,
        "actual_samples": len(samples),
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
