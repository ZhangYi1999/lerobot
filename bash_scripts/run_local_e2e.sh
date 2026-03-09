#!/bin/bash
# End-to-end GR00T N1.5 fine-tuning on LIBERO-10 (train + eval)
# Runs inside clare-training Docker container.
#
# Usage (from clare_rebuttal/):
#   mkdir -p data
#   docker run --gpus all -it --rm \
#     -v $(pwd)/lerobot/src:/app/lerobot/src \
#     -v $(pwd)/peft_lsy/src:/app/peft_lsy/src \
#     -v $(pwd)/data:/runpod-volume \
#     -e WANDB_API_KEY=${WANDB_API_KEY} \
#     -e HF_TOKEN=${HF_TOKEN} \
#     clare-training:latest \
#     bash /app/lerobot/src/lerobot/bash_scripts/run_local_e2e.sh

set -euo pipefail

# --- Config ---
BASE_MODEL="nvidia/GR00T-N1.5-3B"
DATASET="lerobot/libero_10_subtask"
STEPS=1000
BATCH_SIZE=32
SEED=42
OUTPUT_DIR="/runpod-volume/outputs/e2e_finetune"
WANDB_PROJECT="clare"

echo "========================================"
echo " E2E Fine-tuning: GR00T N1.5 on LIBERO-10"
echo " Steps: ${STEPS}, Batch: ${BATCH_SIZE}"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# --- Train ---
echo "Starting training..."
lerobot-train \
  --policy.path=${BASE_MODEL} \
  --dataset.repo_id=${DATASET} \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir=${OUTPUT_DIR} \
  --steps=${STEPS} \
  --batch_size=${BATCH_SIZE} \
  --seed=${SEED} \
  --eval_freq=${STEPS} \
  --save_freq=${STEPS} \
  --log_freq=50 \
  --wandb.enable=true \
  --wandb.project=${WANDB_PROJECT}

echo "Training complete."

# --- Eval ---
CHECKPOINT="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
echo "Starting evaluation from checkpoint: ${CHECKPOINT}"

lerobot-eval \
  --policy.path=${CHECKPOINT} \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --output_dir=${OUTPUT_DIR}/eval \
  --seed=${SEED}

echo "========================================"
echo " E2E Fine-tuning complete!"
echo " Checkpoint: ${OUTPUT_DIR}/checkpoints/last/"
echo " Eval results: ${OUTPUT_DIR}/eval/"
echo "========================================"
