import json
import os
import logging
import click
from datasets import Dataset, load_dataset
from any2json.models.gemini import GeminiModel
from any2json.models.qwen import QwenModel
from tqdm.auto import tqdm
import fastjsonschema

logger = logging.getLogger(__name__)

model_types = {
    "qwen": QwenModel,
    "gemini": GeminiModel,
}


def run_benchmark(model, samples: list[dict]) -> list[dict]:
    results = []
    for sample in tqdm(
        samples,
        total=len(samples),
        desc="Benchmarking",
    ):
        input_data: str | dict = sample["input_data"]

        if isinstance(input_data, dict):
            input_data = json.dumps(input_data)

        schema: dict = sample["schema"]
        correct_answer: dict = sample["output"]

        answer, meta = model.convert_to_json(input_data, schema)

        results.append(
            {
                "input_data": input_data,
                "schema": schema,
                "correct_answer": correct_answer,
                "answer": answer,
                "meta": meta,
            }
        )

    return results


def calculate_metrics(results):
    error = []
    correct = []
    schema_error = []
    for i, result in enumerate(results):
        schema = fastjsonschema.compile(result["schema"])
        try:
            answer = json.loads(result["answer"])

            schema(answer)

            correct_answer = result["correct_answer"]

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
    "--input-file",
    default="data/intermediate/samples.json",
    type=click.Path(exists=True),
    required=True,
    help="Input file to benchmark",
)
@click.option(
    "--model-type",
    default="qwen",
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
def run(input_file, model_type, model_name, output_dir, limit):
    logger.info(f"Benchmarking {input_file} with {model_type} {model_name}")

    model_type = model_types[model_type]
    model = model_type()

    with open(input_file, "r") as f:
        samples = json.load(f)[:limit]

    results = run_benchmark(model, samples)

    metrics = calculate_metrics(results)

    logger.info(f"Metrics: {metrics}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    run()
