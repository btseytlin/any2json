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

echo "Setup complete, running command"
python any2json/training/train.py train --learning_rate=1e-4 --warmup-ratio 0.03 --weight-decay 0.05 --per-device-train-batch-size 6 --per-device-eval-batch-size 2 --gradient-accumulation-steps 4 --num-train-epochs=2 --eval-strategy steps --save-strategy steps --eval-steps 500 --save-steps 500 --logging_steps 50 --report_to wandb --bf16 --dataloader-num-workers 10 --dataloader-prefetch-factor 8 --group-by-length True --dataloader-persistent-workers True --optim adamw_torch_fused --tf32 True --gradient-checkpointing --max-source-length 3096 --max-target-length 2028

