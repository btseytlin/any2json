#!/usr/bin/env bash
set -euo pipefail

cd /code/any2json && source .venv/bin/activate
git fetch --all && git reset --hard origin/main

uv sync && uv pip install -e . && uv -q pip install vllm --torch-backend=auto

echo "Code and dependencies updated"

echo "Downloading checkpoint"
export WANDB_RUN_ID=$(python scripts/wandb_tools.py --quiet get-run-id)

echo "WANDB_RUN_ID: $WANDB_RUN_ID"

# Required environment variables
: ${MODEL_ARTIFACT_ID:?MODEL_ARTIFACT_ID is required}

# Auto-generate model name from artifact ID
# Extract hash and version from artifact ID like btseytlin/any2json/model-31cjr161:v6
ARTIFACT_PART=$(echo "$MODEL_ARTIFACT_ID" | cut -d'/' -f3)
MODEL_HASH=$(echo "$ARTIFACT_PART" | sed 's/model-//' | sed 's/:v[0-9]*//')
MODEL_VERSION=$(echo "$ARTIFACT_PART" | sed 's/.*://')

# Use MODEL_TYPE from environment or default to smollm2
: ${MODEL_TYPE:=smollm2}

MODEL_NAME="${MODEL_TYPE}_${MODEL_HASH}_${MODEL_VERSION}"

echo "Generated MODEL_NAME: $MODEL_NAME"

# Optional environment variables with defaults
: ${BENCHMARK_LIMIT:=500}
: ${RUN_STRUCTURED_OUTPUT:=true}
: ${RUN_NO_STRUCTURED_OUTPUT:=false}
: ${MODEL_PATH_OVERRIDE:=""}
: ${NO_SO_MODEL_PATH_OVERRIDE:=""}

python scripts/wandb_tools.py --run-id $WANDB_RUN_ID download-artifact $MODEL_ARTIFACT_ID --output-root /workspace/models

echo "Downloaded checkpoint"
echo "Setup complete, running commands"

# Determine model path
if [[ -n "$MODEL_PATH_OVERRIDE" ]]; then
    MODEL_PATH="$MODEL_PATH_OVERRIDE"
else
    MODEL_PATH="/workspace/models/${MODEL_ARTIFACT_ID#*/}"
fi

# Structured Output Benchmark
if [[ "$RUN_STRUCTURED_OUTPUT" == "true" ]]; then
    echo "Running structured output benchmark"
    
    python any2json/benchmarks/benchmark.py run --run-id $WANDB_RUN_ID --hf-dataset btseytlin/any2json --split test --model-type vllm_custom --output-dir=benchmark_results \
        --model-kwargs="{\"model_name\": \"$MODEL_PATH\", \"guided_json\": true, \"server_startup_timeout\": 600}" \
        --output-dir /workspace/benchmark_results/${MODEL_NAME}_so \
        --limit $BENCHMARK_LIMIT

    python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/${MODEL_NAME}_so \
        --name any2json-benchmark-${MODEL_NAME}_so --type benchmark_results

    python any2json/benchmarks/benchmark.py metrics /workspace/benchmark_results/${MODEL_NAME}_so

    python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/${MODEL_NAME}_so --incremental --name any2json-benchmark-${MODEL_NAME}_so --type benchmark_results
fi

# No Structured Output Benchmark
if [[ "$RUN_NO_STRUCTURED_OUTPUT" == "true" ]]; then
    echo "Running non-structured output benchmark"
    
    # Use override path if provided, otherwise use default model path
    if [[ -n "$NO_SO_MODEL_PATH_OVERRIDE" ]]; then
        NO_SO_MODEL_PATH="$NO_SO_MODEL_PATH_OVERRIDE"
    else
        NO_SO_MODEL_PATH="$MODEL_PATH"
    fi
    
    python any2json/benchmarks/benchmark.py run --run-id $WANDB_RUN_ID --hf-dataset btseytlin/any2json --split test --model-type vllm_custom --output-dir=benchmark_results \
        --model-kwargs="{\"model_name\": \"$NO_SO_MODEL_PATH\", \"server_startup_timeout\": 600}" \
        --output-dir /workspace/benchmark_results/$MODEL_NAME \
        --limit $BENCHMARK_LIMIT

    python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/$MODEL_NAME \
        --name any2json-benchmark-$MODEL_NAME --type benchmark_results

    python any2json/benchmarks/benchmark.py metrics /workspace/benchmark_results/$MODEL_NAME

    python scripts/wandb_tools.py --run-id $WANDB_RUN_ID upload-directory /workspace/benchmark_results/$MODEL_NAME --incremental --name any2json-benchmark-$MODEL_NAME --type benchmark_results
fi

echo "Benchmark completed"
