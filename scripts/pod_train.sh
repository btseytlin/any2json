#!/usr/bin/env bash
set -euo pipefail

cd /code/any2json && source .venv/bin/activate
git fetch --all && git reset --hard origin/main

uv sync && uv pip install -e .

cd /workspace

echo "Code and dependencies updated"

echo "Setup complete, running command"

# For A40 GPU, gemma3 270m, 3096 max seq len

export BATCH_SIZE=3
export NUM_EPOCHS=2
export MODEL=google/gemma-3-270m
# export MODEL=HuggingFaceTB/SmolLM2-135M

python /code/any2json/any2json/training/train.py train \
    --model-name=$MODEL \
    --max-sequence-length 3096 \
    --group-by-length True --length-column-name length \
    --learning_rate=5e-5 --warmup-ratio 0.03 --weight-decay 0.01 \
    --eval-on-start True \
    --eval-strategy steps  --report_to wandb --save-strategy steps --save-total-limit 3 \
    --eval-steps 0.05 --save-steps 0.2 \
    --optim adamw_torch_fused --tf32 True --bf16 \
    --dataloader-num-workers 4 --dataloader-prefetch-factor 1 \
    --gradient-checkpointing \
    --num-train-epochs=$NUM_EPOCHS \
    --per-device-train-batch-size $BATCH_SIZE --per-device-eval-batch-size $BATCH_SIZE \
    --augment \
    --augment-val