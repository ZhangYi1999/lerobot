#!/bin/bash
set -e

# ==============================================================================
# Stage 3: Train CLARE discriminators per task
# Requires a converted CLARE checkpoint (from convert_lora_to_clare.sh).
# Trains one discriminator per task sequentially.
# ==============================================================================

# ===== Configuration =====
PRETRAINED_PATH="outputs/train/dit_fft_pretraining_v1_s1000/checkpoints/last/pretrained_model"
BASE_OUTPUT="outputs/lora_to_clare"
CLARE_CHECKPOINT="${BASE_OUTPUT}/clare_converted/adapter/default"

DISC_STEPS=2000
DISC_LOG_FREQ=50
DISC_BATCH_SIZE=32
SEED=1000

DATASETS=(
    "real_0_put_bowl_filtered"
    "real_1_stack_bowls_filtered"
    # "real_2_put_moka_pot_filtered"
    # "real_3_close_drawer_filtered"
)

# ===== Validate =====
if [ ! -d "${CLARE_CHECKPOINT}" ]; then
    echo "ERROR: CLARE checkpoint not found at ${CLARE_CHECKPOINT}"
    echo "  Run convert_lora_to_clare.sh first"
    exit 1
fi

N_TASKS=${#DATASETS[@]}
DISC_OUTPUT="${BASE_OUTPUT}/clare_final"

echo "=========================================="
echo "CLARE Discriminator Training (DiT)"
echo "  Tasks:      ${N_TASKS}"
echo "  Checkpoint: ${CLARE_CHECKPOINT}"
echo "  Output:     ${DISC_OUTPUT}"
echo "  Steps/task: ${DISC_STEPS}"
echo "=========================================="

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"

    echo ""
    echo "--- Discriminator for task ${i}: ${DATASET} ---"

    python -m lerobot.scripts.clare.clare \
        --phase=discriminator_only \
        --peft_weight_path="${CLARE_CHECKPOINT}" \
        --discriminator_task_id=${i} \
        --output_dir="${DISC_OUTPUT}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --policy.type=dit \
        --policy.pretrained_path="${PRETRAINED_PATH}" \
        --policy.push_to_hub=false \
        --batch_size=${DISC_BATCH_SIZE} \
        --num_workers=16 \
        --seed=${SEED} \
        --eval_freq=0 \
        --train_discriminators_steps=${DISC_STEPS} \
        --train_discriminators_log_freq=${DISC_LOG_FREQ} \
        --train_discriminators_save_freq=${DISC_STEPS} \
        --wandb.enable=true \
        --wandb.disable_artifact=true \
        --wandb.project=clare_rebuttal \
        --wandb.entity=470620104-technical-university-of-munich
done

echo ""
echo "=========================================="
echo "Discriminator training complete!"
echo "  Final model: ${DISC_OUTPUT}/"
echo "=========================================="
