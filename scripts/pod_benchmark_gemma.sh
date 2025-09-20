#!/usr/bin/env bash
set -euo pipefail

cd /code/any2json && source .venv/bin/activate
git fetch --all && git reset --hard origin/main

uv sync && uv pip install -e . && uv -q pip install vllm --torch-backend=auto


echo "Code and dependencies updated"

echo "Downloading checkpoint"
export WANDB_RUN_ID=$(python scripts/wandb_tools.py --quiet get-run-id)

echo "WANDB_RUN_ID: $WANDB_RUN_ID"

export ARTIFACT_ID=btseytlin/any2json/model-yaurre5k:v1
export MODEL_NAME=gemma270m_epoch1

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID download-artifact --artifact-id $ARTIFACT_ID --output-root /workspace/models

echo "Downloaded checkpoint"

echo "Setup complete, running commands"

# Structured Output

python any2json/benchmarks/benchmark.py run --hf-dataset btseytlin/any2json --split test --model-type vllm_custom --output-dir=benchmark_results  \
    --model-kwargs='{"model_name": "/workspace/models/any2json_gemma270m:epoch1", "guided_json": true, "server_startup_timeout": 600}' \
    --output-dir /workspace/benchmark_results/gemma270m_epoch1_so \
    --limit 500

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/gemma270m_epoch1_so \
    --name any2json-benchmark-gemma270m_epoch1_so --type benchmark_results

python any2json/benchmarks/benchmark.py metrics /workspace/benchmark_results/gemma270m_epoch1_so

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/gemma270m_epoch1_so --incremental --name any2json-benchmark-gemma270m_epoch1_so --type benchmark_results

# No Structured Output

python any2json/benchmarks/benchmark.py run --hf-dataset btseytlin/any2json --split test --model-type vllm_custom --output-dir=benchmark_results  \
    --model-kwargs='{"model_name": "/workspace/models/any2json_gemma270m:epoch1", "server_startup_timeout": 600}' \
    --output-dir /workspace/benchmark_results/gemma270m_epoch1 \
    --limit 500

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/gemma270m_epoch1 \
    --name any2json-benchmark-gemma270m_epoch1 --type benchmark_results

python any2json/benchmarks/benchmark.py metrics /workspace/benchmark_results/gemma270m_epoch1

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/gemma270m_epoch1 --incremental --name any2json-benchmark-gemma270m_epoch1 --type benchmark_results
