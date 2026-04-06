#!/bin/bash
set -e

# Full fine-tuning with frozen vision encoder
# GPU: 1x H100-94GB, estimated ~52-57GB VRAM

# ===== Configuration =====
PRETRAINED_PATH="lerobot/pi0_base"
POLICY_TYPE="pi0"
METHOD="fullft_fv"
STEPS=20000
SAVE_FREQ=4000
LOG_FREQ=100
BATCH_SIZE=16
NUM_WORKERS=8
LR=2.5e-5
DECAY_LR=2.5e-6
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEBUG="${DEBUG:-false}"

# Debug mode overrides
if [ "$DEBUG" = "true" ]; then
    STEPS=1000
    SAVE_FREQ=1000
    LOG_FREQ=1
    PUSH_TO_HUB=false
    WANDB_ENABLE=true
    WANDB_PROJECT="debug"
else
    PUSH_TO_HUB=true
    WANDB_ENABLE=true
    WANDB_PROJECT="clare_rebuttal"
fi

DATASETS=(
    "real_0_put_bowl_filtered_consolidated"
    "real_1_stack_bowls_filtered_consolidated"
    "real_2_put_moka_pot_filtered_consolidated"
    "real_3_close_drawer_filtered_consolidated"
    "real_4_put_lego_into_drawer_filtered_consolidated"
)

for DATASET in "${DATASETS[@]}"; do
    JOB_NAME="${POLICY_TYPE}_${METHOD}_${DATASET}"
    REPO_ID="continuallearning/${JOB_NAME}"
    OUTPUT_DIR="./outputs/train/${JOB_NAME}"

    echo "=========================================="
    echo "Full FT (frozen vision) ${POLICY_TYPE} on: ${DATASET}"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "=========================================="

    # Skip if already completed
    FINAL_CKPT="${OUTPUT_DIR}/checkpoints/$(printf '%05d' ${STEPS})"
    if [ -d "${FINAL_CKPT}" ]; then
        echo "  Already completed (found ${FINAL_CKPT}). Skipping."
        continue
    fi

    # Resume from last checkpoint if available
    RESUME_ARGS=""
    LAST_CKPT="${OUTPUT_DIR}/checkpoints/last"
    if [ -d "${LAST_CKPT}" ] || [ -L "${LAST_CKPT}" ]; then
        CONFIG_FILE="${LAST_CKPT}/pretrained_model/train_config.json"
        if [ -f "${CONFIG_FILE}" ]; then
            echo "  Resuming from checkpoint: ${LAST_CKPT}"
            RESUME_ARGS="--resume --config_path=${CONFIG_FILE}"
        fi
    fi

    accelerate launch --mixed_precision=${MIXED_PRECISION} \
        -m lerobot.scripts.lerobot_train \
        --job_name="${JOB_NAME}" \
        --output_dir="${OUTPUT_DIR}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --dataset.image_transforms.enable=true \
        --dataset.image_transforms.max_num_transforms=3 \
        --dataset.image_transforms.random_order=false \
        --policy.type=${POLICY_TYPE} \
        --policy.pretrained_path="${PRETRAINED_PATH}" \
        --policy.dtype=bfloat16 \
        --policy.gradient_checkpointing=true \
        --policy.compile_model=true \
        --policy.freeze_vision_encoder=true \
        --policy.optimizer_lr=${LR} \
        --policy.scheduler_warmup_steps=500 \
        --policy.scheduler_decay_steps=${STEPS} \
        --policy.scheduler_decay_lr=${DECAY_LR} \
        --policy.chunk_size=10 \
        --policy.n_action_steps=10 \
        --policy.push_to_hub=${PUSH_TO_HUB} \
        --policy.repo_id="${REPO_ID}" \
        --batch_size=${BATCH_SIZE} \
        --num_workers=${NUM_WORKERS} \
        --steps=${STEPS} \
        --eval_freq=0 \
        --save_freq=${SAVE_FREQ} \
        --log_freq=${LOG_FREQ} \
        --wandb.enable=${WANDB_ENABLE} \
        --wandb.disable_artifact=true \
        --wandb.project=${WANDB_PROJECT} \
        --wandb.entity=470620104-technical-university-of-munich \
        ${RESUME_ARGS}
done

echo "All ${POLICY_TYPE} ${METHOD} fine-tuning runs completed!"
