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

python any2json/training/train.py estimate-lengths 

poweroff









