#!/usr/bin/env bash
set -euo pipefail

cd /code/any2json && source .venv/bin/activate

git fetch --all && git reset --hard origin/main
echo "Code updated"

echo "Downloading checkpoint"
export WANDB_RUN_ID=$(python scripts/wandb_tools.py --quiet get-run-id)

echo "WANDB_RUN_ID: $WANDB_RUN_ID"

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID download-model --model-id btseytlin/model-registry/any2json_gemma270m:latest --output-root /workspace/models

echo "Downloaded checkpoint"

echo "Setup complete, running commands"

python any2json/benchmarks/benchmark.py run --hf-dataset btseytlin/any2json --split test --model-type vllm_custom --output-dir=benchmark_results --model-kwargs='{"model_name": "/workspace/models/any2json_gemma270m:latest", "guided_json": true, "server_startup_timeout": 600}' --output-dir /workspace/benchmark_results/gemma270m_so --limit 500
python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/gemma270m_so --name any2json-benchmark-gemma270m-so --type benchmark_results

python any2json/benchmarks/benchmark.py metrics /workspace/benchmark_results/gemma270m_so

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/gemma270m_so --incremental --name any2json-benchmark-gemma270m-so --type benchmark_results
