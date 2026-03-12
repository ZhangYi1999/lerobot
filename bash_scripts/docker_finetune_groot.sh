#!/bin/bash
set -e

# Docker wrapper for finetune_groot.sh
# Run from workspace root (XLerobot_workspace/):
#   bash lerobot/bash_scripts/docker_finetune_groot.sh

# ---- Configure these ----
IMAGE="ghcr.io/zhangyi1999/clare-training:latest"
HF_CACHE_DIR="${HOME}/.cache/huggingface"
# --------------------------

WORKSPACE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

docker run --gpus all --rm \
    -v "${WORKSPACE_DIR}/lerobot:/app/lerobot" \
    -v "${WORKSPACE_DIR}/peft_lsy:/app/peft_lsy" \
    -v "${WORKSPACE_DIR}/outputs:/app/outputs" \
    -v "${HF_CACHE_DIR}:/runpod-volume/huggingface" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    "${IMAGE}" \
    bash /app/lerobot/bash_scripts/finetune_groot.sh
