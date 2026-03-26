#!/bin/bash
set -e

# ==============================================================================
# Train CLARE discriminators per task (one at a time)
# Requires the merged CLARE checkpoint from convert_lora_to_clare.sh.
# ==============================================================================

# ===== Configuration =====
DISC_STEPS=10000
DISC_SAVE_FREQ=10000
DISC_LOG_FREQ=50
DISC_BATCH_SIZE=256
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export REUSE_PRETRAINED_NORMALIZATION="${REUSE_PRETRAINED_NORMALIZATION:-true}"
START_TASK="${START_TASK:-0}"
DEBUG="${DEBUG:-false}"
SEED=1000

# Debug mode overrides
if [ "$DEBUG" = "true" ]; then
    DISC_STEPS=10
    DISC_SAVE_FREQ=10
    DISC_LOG_FREQ=1
    DISC_BATCH_SIZE=256
    PUSH_TO_HUB=false
    WANDB_ENABLE=true
    WANDB_PROJECT="debug"
else
    PUSH_TO_HUB=true
    WANDB_ENABLE=true
    WANDB_PROJECT="clare_rebuttal"
fi

# Normalization
NORM_MODE="${NORM_MODE:-union}"
NORM_STATS_FILE="${NORM_STATS_FILE:-configs/union_stats.json}"

# Pretrained backbone
PRETRAINED_PATH="continuallearning/dit_fft_pretraining_v2_lerobot30_seed1000"

# Merged CLARE checkpoint (from convert_lora_to_clare.sh)
CLARE_CHECKPOINT="outputs/lora_to_clare/clare_converted_v2/adapter"

DATASETS=(
    "real_0_put_bowl_filtered"
    "real_1_stack_bowls_filtered"
    "real_2_put_moka_pot_filtered"
    "real_3_close_drawer_filtered"
    "real_4_put_lego_into_drawer_filtered"
)

# ===== Validate =====
if [ ! -d "${CLARE_CHECKPOINT}" ]; then
    echo "ERROR: CLARE checkpoint not found at ${CLARE_CHECKPOINT}"
    echo "  Run convert_lora_to_clare.sh first"
    exit 1
fi

N_TASKS=${#DATASETS[@]}

echo "=========================================="
echo "CLARE Discriminator Training (DiT)"
echo "  Tasks:      ${N_TASKS}"
echo "  Checkpoint: ${CLARE_CHECKPOINT}"
echo "  Steps/task: ${DISC_STEPS}"
echo "=========================================="

for i in "${!DATASETS[@]}"; do
    if [ "$i" -lt "$START_TASK" ]; then
        echo "Skipping task ${i} (START_TASK=${START_TASK})"
        continue
    fi

    DATASET="${DATASETS[$i]}"
    REPO_ID="continuallearning/dit_posttrainv2_clare_discriminator_dit_${DATASET}_seed${SEED}"
    JOB_NAME="${REPO_ID#continuallearning/}"
    OUTPUT_DIR="./outputs/train/${JOB_NAME}"

    # ===== Normalization Source =====
    if [ "$NORM_MODE" = "union" ]; then
        export NORM_STATS_FILE="${NORM_STATS_FILE}"
        unset NORM_CHECKPOINT_PATH
    fi

    # CLARE-specific env vars
    export CLARE_PHASE="discriminator_only"
    export PEFT_WEIGHT_PATH="${CLARE_CHECKPOINT}"
    export DISCRIMINATOR_TASK_ID=${i}
    export TRAIN_DISCRIMINATORS_STEPS=${DISC_STEPS}
    export TRAIN_DISCRIMINATORS_LOG_FREQ=${DISC_LOG_FREQ}
    export TRAIN_DISCRIMINATORS_SAVE_FREQ=${DISC_SAVE_FREQ}

    echo ""
    echo "--- Discriminator for task ${i}: ${DATASET} ---"

    accelerate launch --mixed_precision=${MIXED_PRECISION} \
        -m lerobot.scripts.clare.clare \
        --use_policy_training_preset=false \
        --job_name="${JOB_NAME}" \
        --output_dir="${OUTPUT_DIR}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --policy.type=dit \
        --policy.pretrained_path="${PRETRAINED_PATH}" \
        --policy.push_to_hub=${PUSH_TO_HUB} \
        --policy.repo_id="${REPO_ID}" \
        --batch_size=${DISC_BATCH_SIZE} \
        --num_workers=8 \
        --seed=${SEED} \
        --eval_freq=0 \
        --wandb.enable=${WANDB_ENABLE} \
        --wandb.disable_artifact=true \
        --wandb.project=${WANDB_PROJECT} \
        --wandb.entity=470620104-technical-university-of-munich
done

echo ""
echo "=========================================="
echo "All discriminator training complete!"
echo "=========================================="
