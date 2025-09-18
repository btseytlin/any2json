#!/usr/bin/env bash
set -euo pipefail

cd /code/any2json && source .venv/bin/activate
git fetch --all && git reset --hard origin/main

uv sync && uv pip install -e .

cd /workspace

echo "Code and dependencies updated"

echo "Setup complete, running command"
#python any2json/training/train.py train --learning_rate=1e-4 --warmup-ratio 0.03 --weight-decay 0.05 --per-device-train-batch-size 6 --per-device-eval-batch-size 2 --gradient-accumulation-steps 4 --num-train-epochs=2 --eval-strategy steps --save-strategy steps --eval-steps 500 --save-steps 500 --logging_steps 50 --report_to wandb --bf16 --dataloader-num-workers 10 --dataloader-prefetch-factor 8 --group-by-length True --dataloader-persistent-workers True --optim adamw_torch_fused --tf32 True --gradient-checkpointing --max-source-length 3096 --max-target-length 2028

BATCH_SIZE=10
python /code/any2json/any2json/training/train.py train --learning_rate=5e-5 --warmup-ratio 0.03 --weight-decay 0.01 --eval-strategy steps  --report_to wandb --bf16 --dataloader-num-workers 4 --dataloader-prefetch-factor 1 --group-by-length True --length-column-name length --optim adamw_torch_fused --tf32 True --gradient-checkpointing --save-strategy steps --save-total-limit 3 --num-train-epochs=5 --per-device-train-batch-size $BATCH_SIZE --gradient-accumulation-steps 1 --per-device-eval-batch-size $BATCH_SIZE --eval-steps 4000 --save-steps 4000 --max-sequence-length 3096 --model-name=google/gemma-3-270m
