#!/usr/bin/env bash
set -euo pipefail

cd /code/any2json && source .venv/bin/activate
git fetch --all && git reset --hard origin/main

uv sync && uv pip install -e .

cd /workspace

echo "Code and dependencies updated"

echo "Setup complete, running command"

export BATCH_SIZE=4
export NUM_EPOCHS=2
export MODEL=HuggingFaceTB/SmolLM2-135M

export WANDB_ARTIFACT_ID=btseytlin/any2json/model-i7dfz4h2:v6

if [ -n "$WANDB_ARTIFACT_ID" ]; then
    echo "Downloading artifact: $WANDB_ARTIFACT_ID"
    python /code/any2json/scripts/wandb_tools.py download-artifact $WANDB_ARTIFACT_ID --output-root /workspace/models

    ARTIFACT_PART=$(echo "$WANDB_ARTIFACT_ID" | cut -d'/' -f3)

    MODEL=/workspace/models/$ARTIFACT_PART
    echo "Using model: $MODEL"
fi

python /code/any2json/any2json/training/train.py train \
    --model-name=$MODEL \
    --max-sequence-length 3072 \
    --group-by-length True --length-column-name length \
    --learning_rate=5e-5 --warmup-ratio 0.05 \
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
