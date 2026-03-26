#!/bin/bash
set -e

# ===== Configuration =====
STEPS=20000
SAVE_FREQ=20000
LOG_FREQ=100
DISC_STEPS=10000
DISC_LOG_FREQ=1
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export REUSE_PRETRAINED_NORMALIZATION="${REUSE_PRETRAINED_NORMALIZATION:-true}"
START_TASK="${START_TASK:-0}"   # Set to resume from middle, e.g. START_TASK=1
DEBUG="${DEBUG:-false}"
SEED=1000

# Debug mode overrides
if [ "$DEBUG" = "true" ]; then
    STEPS=10
    DISC_STEPS=10
    SAVE_FREQ=10
    LOG_FREQ=1
    DISC_LOG_FREQ=1
    BATCH_SIZE=256
    PUSH_TO_HUB=false
    WANDB_ENABLE=true
    WANDB_PROJECT="debug"
else
    BATCH_SIZE=256
    PUSH_TO_HUB=true
    WANDB_ENABLE=true
    WANDB_PROJECT="clare_rebuttal"
fi

# CLARE adapter config
PEFT_CFG="configs/peft/clare_dit_cond_proj"

# Normalization source control:
#   "pretrained" (default) — each task uses normalization from its own CHECKPOINTS[i]
#   "first"                — task 0 loads from dataset; task 1+ reuse normalization from task 0's output checkpoint
#   "union"                — all tasks use pre-computed union stats from NORM_STATS_FILE
NORM_MODE="${NORM_MODE:-union}"
NORM_STATS_FILE="${NORM_STATS_FILE:-configs/union_stats.json}"

# The base pretrained model (stays constant — CLARE modifies adapters, not the base model)
PRETRAINED_PATH="continuallearning/dit_fft_pretraining_v2_lerobot30_seed1000"

DATASETS=(
    "real_0_put_bowl_filtered_consolidated"
    "real_1_stack_bowls_filtered_consolidated"
    "real_2_put_moka_pot_filtered_consolidated"
    "real_3_close_drawer_filtered_consolidated"
    "real_4_put_lego_into_drawer_filtered_consolidated"
)

# ===== Training Loop =====
PREV_OUTPUT_DIR=""

for i in "${!DATASETS[@]}"; do
    if [ "$i" -lt "$START_TASK" ]; then
        echo "Skipping task ${i} (START_TASK=${START_TASK})"
        # Still track the output dir for adapter chaining
        DATASET="${DATASETS[$i]}"
        REPO_ID="continuallearning/dit_posttrainv2_clare_dit_cond_proj_${DATASET}_seed${SEED}"
        JOB_NAME="${REPO_ID#continuallearning/}"
        PREV_OUTPUT_DIR="./outputs/train/${JOB_NAME}"
        continue
    fi

    DATASET="${DATASETS[$i]}"
    REPO_ID="continuallearning/dit_posttrainv2_clare_dit_cond_proj_${DATASET}_seed${SEED}"

    # Local job name / output dir mirrors the hub repo_id suffix
    JOB_NAME="${REPO_ID#continuallearning/}"
    OUTPUT_DIR="./outputs/train/${JOB_NAME}"

    # ===== PEFT env vars: task 0 uses config, subsequent tasks load previous adapter =====
    if [ "$i" -eq 0 ]; then
        export PEFT_CFG_PATH="${PEFT_CFG}"
        unset PEFT_WEIGHT_PATH
    else
        export PEFT_WEIGHT_PATH="${PREV_OUTPUT_DIR}/checkpoints/last/adapter"
        unset PEFT_CFG_PATH
    fi

    # ===== Normalization Source =====
    if [ "$NORM_MODE" = "union" ]; then
        export NORM_STATS_FILE="${NORM_STATS_FILE}"
        unset NORM_CHECKPOINT_PATH
        echo "NORM_MODE=union, task ${i}: using union stats from ${NORM_STATS_FILE}"
    elif [ "$NORM_MODE" = "first" ]; then
        unset NORM_STATS_FILE
        if [ "$i" -eq 0 ]; then
            unset NORM_CHECKPOINT_PATH
            echo "NORM_MODE=first, task 0: loading normalization from dataset"
        else
            TASK0_REPO_ID="continuallearning/dit_posttrainv2_clare_dit_${DATASETS[0]}_seed${SEED}"
            export NORM_CHECKPOINT_PATH="${TASK0_REPO_ID}"
            echo "NORM_MODE=first, task ${i}: reusing normalization from ${TASK0_REPO_ID}"
        fi
    else
        unset NORM_CHECKPOINT_PATH
        unset NORM_STATS_FILE
    fi

    # CLARE-specific env vars
    export CLARE_PHASE="full"
    export TRAIN_DISCRIMINATORS_STEPS=${DISC_STEPS}
    export TRAIN_DISCRIMINATORS_LOG_FREQ=${DISC_LOG_FREQ}
    export EXPAND_THRESHOLD=0.0
    export AT_LEAST_EXPAND=shallowest

    echo "=========================================="
    echo "CLARE DiT: ${DATASET} (task=${i}, seed=${SEED})"
    echo "  Pretrained: ${PRETRAINED_PATH}"
    echo "  PEFT_CFG_PATH: ${PEFT_CFG_PATH:-unset}"
    echo "  PEFT_WEIGHT_PATH: ${PEFT_WEIGHT_PATH:-unset}"
    echo "  To:         ${REPO_ID}"
    echo "=========================================="

    accelerate launch --mixed_precision=${MIXED_PRECISION} \
        -m lerobot.scripts.clare.clare \
        --use_policy_training_preset=false \
        --job_name="${JOB_NAME}" \
        --output_dir="${OUTPUT_DIR}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --dataset.image_transforms.enable=true \
        --policy.type=dit \
        --policy.pretrained_path="${PRETRAINED_PATH}" \
        --policy.push_to_hub=${PUSH_TO_HUB} \
        --policy.repo_id="${REPO_ID}" \
        --batch_size=${BATCH_SIZE} \
        --num_workers=8 \
        --steps=${STEPS} \
        --seed=${SEED} \
        --eval_freq=0 \
        --save_freq=${SAVE_FREQ} \
        --log_freq=${LOG_FREQ} \
        --wandb.enable=${WANDB_ENABLE} \
        --wandb.disable_artifact=true \
        --wandb.project=${WANDB_PROJECT} \
        --wandb.entity=470620104-technical-university-of-munich

    PREV_OUTPUT_DIR="${OUTPUT_DIR}"
done

echo "All CLARE DiT posttrain runs completed!"
