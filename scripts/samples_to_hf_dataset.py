import click
import json
import datasets

from any2json.models.qwen import QwenModel


def to_hf_dataset_entries(samples: list[dict]) -> list[dict]:
    results = []
    for sample in samples:
        conversation = [
            {
                "role": "system",
                "content": QwenModel.system_prompt,
            }
        ]

        conversation.append(
            {
                "role": "user",
                "content": QwenModel.make_prompt(
                    sample["input_data"],
                    sample["schema"],
                ),
            }
        )
        conversation.append(
            {
                "role": "assistant",
                "content": json.dumps(sample["output"], indent=2),
            }
        )

        results.append(
            {
                "messages": conversation,
            }
        )

    return results


@click.command()
@click.option(
    "--input-file",
    default="data/intermediate/samples.json",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output-dir",
    default="data/final/hf_sft_dataset",
    type=click.Path(),
    required=True,
)
def run(
    input_file: str,
    output_dir: str,
):
    with open(input_file, "r") as f:
        samples = json.load(f)

    dataset = datasets.Dataset.from_list(to_hf_dataset_entries(samples))

    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    run()
