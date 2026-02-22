#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_PROJECT="removing-layer-norm-lora"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/venv/bin/python"

CONFIGS=(
    gpt2-large_lora_lr1e-4
    gpt2-xl_lora_lr1e-4
)

for cfg in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Starting run: ${cfg}  $(date)"
    echo "=========================================="
    $PYTHON "${SCRIPT_DIR}/train_lora.py" --config "${cfg}" 2>&1 | tee "${SCRIPT_DIR}/lora_${cfg}.log"
    echo "Finished: ${cfg}  $(date)"
    echo ""
done

echo "All scale runs complete.  $(date)"
