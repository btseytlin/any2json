#!/usr/bin/env bash
set -euo pipefail

apt-get update && apt-get install -y git curl nvtop tmux

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p /workspace
cd /workspace

if [ ! -d any2json ]; then
    git clone --filter=blob:none  https://github.com/btseytlin/any2json.git
fi

cd any2json
git pull && git fetch && git reset --hard origin/main

export UV_LINK_MODE=copy
uv sync && source .venv/bin/activate && uv pip install -e .

uv -q pip install vllm --torch-backend=auto

echo "Downloading checkpoint"
python scripts/wandb_tools.py download-model --model-id btseytlin/model-registry/any2json_gemma270m:latest

echo "Setup complete, running command"

python any2json/benchmarks/benchmark.py run --hf-dataset btseytlin/any2json --split test --model-type vllm_custom --output-dir=benchmark_results \ 
    --model-kwargs='{"model_name": "./models/any2json_gemma270m:latest", "guided_json": true, "server_startup_timeout": 600}' --output-dir benchmark_results/gemma270m_so --limit 500
python scripts/wandb_tools.py upload-directory benchmark_results/gemma270m_so --name any2json-benchmark-gemma270m-so --type benchmark_results

python any2json/benchmarks/benchmark.py metrics benchmark_results/gemma270m_so

python scripts/wandb_tools.py upload-directory benchmark_results/gemma270m_so --incremental --name any2json-benchmark-gemma270m-so --type benchmark_results
