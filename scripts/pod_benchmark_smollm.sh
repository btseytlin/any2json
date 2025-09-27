#!/usr/bin/env bash
set -euo pipefail

cd /code/any2json && source .venv/bin/activate
git fetch --all && git reset --hard origin/main

uv sync && uv pip install -e . && uv -q pip install vllm --torch-backend=auto


echo "Code and dependencies updated"

echo "Downloading checkpoint"
export WANDB_RUN_ID=$(python scripts/wandb_tools.py --quiet get-run-id)

echo "WANDB_RUN_ID: $WANDB_RUN_ID"

export MODEL_ID=btseytlin/any2json/model-f2yvr0zy:v6
export MODEL_ID_NAME=model-f2yvr0zy:v6
export MODEL_NAME=smollm2_f2yvr0zy_v6

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID download-artifact $MODEL_ID --output-root /workspace/models

echo "Downloaded checkpoint"

echo "Setup complete, running commands"

python any2json/benchmarks/benchmark.py run --run-id $WANDB_RUN_ID --hf-dataset btseytlin/any2json --split test --model-type vllm_custom --output-dir=benchmark_results  \
    --model-kwargs='{"model_name": "/workspace/models/model-f2yvr0zy:v6", "guided_json": true, "server_startup_timeout": 600}' \
    --output-dir /workspace/benchmark_results/smollm2_f2yvr0zy_v6_so \
    --limit 500

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/smollm2_f2yvr0zy_v6_so \
    --name any2json-benchmark-smollm2_f2yvr0zy_v6_so --type benchmark_results

python any2json/benchmarks/benchmark.py metrics /workspace/benchmark_results/smollm2_f2yvr0zy_v6_so

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/smollm2_f2yvr0zy_v6_so --incremental --name any2json-benchmark-smollm2_f2yvr0zy_v6_so --type benchmark_results
