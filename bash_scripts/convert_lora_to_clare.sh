#!/bin/bash
set -e

# ==============================================================================
# Stage 2: Convert LoRA checkpoints to CLARE format
# Takes N LoRA adapter dirs and produces a single CLARE checkpoint with
# adapters populated and discriminators randomly initialized.
# ==============================================================================

# ===== Configuration =====
PRETRAINED_PATH="outputs/train/dit_fft_pretraining_v1_s1000/checkpoints/last/pretrained_model"
CLARE_CONFIG_PATH="configs/peft/clare_dit"
BASE_OUTPUT="outputs/lora_to_clare"

DATASETS=(
    "real_0_put_bowl_filtered"
    "real_1_stack_bowls_filtered"
    # "real_2_put_moka_pot_filtered"
    # "real_3_close_drawer_filtered"
)

# ===== Build list of LoRA checkpoint dirs =====
LORA_CHECKPOINT_DIRS=""
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    LORA_DIR="${BASE_OUTPUT}/lora/task_${i}_${DATASET}/checkpoints/last/adapter"
    if [ ! -d "${LORA_DIR}" ]; then
        echo "ERROR: LoRA checkpoint not found at ${LORA_DIR}"
        echo "  Run LoRA training first (e.g., lora_finetune_dit.sh per task)"
        exit 1
    fi
    LORA_CHECKPOINT_DIRS="${LORA_CHECKPOINT_DIRS} ${LORA_DIR}"
done

N_TASKS=${#DATASETS[@]}
CLARE_OUTPUT="${BASE_OUTPUT}/clare_converted"

echo "=========================================="
echo "Convert LoRA → CLARE"
echo "  Tasks:        ${N_TASKS}"
echo "  CLARE config: ${CLARE_CONFIG_PATH}"
echo "  Output:       ${CLARE_OUTPUT}"
echo "=========================================="

# Use the first dataset just for metadata (ds_meta)
python -m lerobot.scripts.clare.convert_lora_to_clare \
    --lora_checkpoint_dirs ${LORA_CHECKPOINT_DIRS} \
    --clare_config_path="${CLARE_CONFIG_PATH}" \
    --output_dir="${CLARE_OUTPUT}" \
    --policy.type=dit \
    --policy.pretrained_path="${PRETRAINED_PATH}" \
    --policy.push_to_hub=false \
    --dataset.repo_id="continuallearning/${DATASETS[0]}" \
    --eval_freq=0

echo ""
echo "Conversion complete: ${CLARE_OUTPUT}"
echo "Next: run clare_finetune_dit.sh to train discriminators"
