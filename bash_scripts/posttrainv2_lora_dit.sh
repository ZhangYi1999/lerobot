#!/bin/bash
set -e

# ===== Configuration =====
# Single-task LoRA fine-tuning (no gradient accumulation) with image transforms.
PRETRAINED_PATH="continuallearning/dit_fft_pretraining_v2_lerobot30_seed1000"
STEPS=40000
SAVE_FREQ=40000
LOG_FREQ=100
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export REUSE_PRETRAINED_NORMALIZATION="${REUSE_PRETRAINED_NORMALIZATION:-true}"
export MERGE_LORA_ADAPTER="${MERGE_LORA_ADAPTER:-false}"
START_TASK="${START_TASK:-0}"   # Set to resume from middle, e.g. START_TASK=1
DEBUG="${DEBUG:-false}"
SEED=1000

# Debug mode overrides
if [ "$DEBUG" = "true" ]; then
    STEPS=10
    SAVE_FREQ=10
    LOG_FREQ=1
    BATCH_SIZE=128
    PUSH_TO_HUB=false
    WANDB_ENABLE=true
    WANDB_PROJECT="debug"
else
    BATCH_SIZE=128
    PUSH_TO_HUB=true
    WANDB_ENABLE=true
    WANDB_PROJECT="clare_rebuttal"
fi

# LoRA adapter config
LORA_CONFIG="${LORA_CONFIG:-dit_all}"  # Options: dit_encoder, dit_decoder, dit_all, dit_all_decoder
export PEFT_CONFIG_PATH="configs/lora/${LORA_CONFIG}"

if [ ! -f "${PEFT_CONFIG_PATH}/adapter_config.json" ]; then
    echo "ERROR: adapter_config.json not found at ${PEFT_CONFIG_PATH}"
    exit 1
fi
echo "Using LoRA config: ${PEFT_CONFIG_PATH}/adapter_config.json"

# Normalization source control:
#   "pretrained" (default) — each task uses normalization from its own pretrained checkpoint
#   "first"                — task 0 loads from dataset; task 1+ reuse normalization from task 0's output checkpoint
#   "union"                — all tasks use pre-computed union stats from NORM_STATS_FILE
NORM_MODE="${NORM_MODE:-union}"
NORM_STATS_FILE="${NORM_STATS_FILE:-configs/union_stats.json}"

DATASETS=(
    "real_0_put_bowl_filtered"
    "real_1_stack_bowls_filtered"
    "real_2_put_moka_pot_filtered"
    "real_3_close_drawer_filtered"
    "real_4_put_lego_into_drawer_filtered"
)

# ===== Training Loop =====
for i in "${!DATASETS[@]}"; do
    if [ "$i" -lt "$START_TASK" ]; then
        echo "Skipping task ${i} (START_TASK=${START_TASK})"
        continue
    fi

    DATASET="${DATASETS[$i]}"
    REPO_ID="continuallearning/dit_posttrainv2_baseline_lora_${LORA_CONFIG}_${DATASET}_seed${SEED}"

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
            TASK0_REPO_ID="continuallearning/dit_posttrainv2_baseline_lora_${LORA_CONFIG}_${DATASETS[0]}_seed${SEED}"
            export NORM_CHECKPOINT_PATH="${TASK0_REPO_ID}"
            echo "NORM_MODE=first, task ${i}: reusing normalization from ${TASK0_REPO_ID}"
        fi
    else
        unset NORM_CHECKPOINT_PATH
        unset NORM_STATS_FILE
    fi

    # Local job name / output dir mirrors the hub repo_id suffix
    JOB_NAME="${REPO_ID#continuallearning/}"

    echo "=========================================="
    echo "LoRA (${LORA_CONFIG}) DiT: ${DATASET} (task=${i}, seed=${SEED})"
    echo "  From: ${PRETRAINED_PATH}"
    echo "  To:   ${REPO_ID}"
    echo "  BS: 128, image transforms enabled"
    echo "=========================================="

    accelerate launch --mixed_precision=${MIXED_PRECISION} \
        -m lerobot.scripts.lerobot_train \
        --job_name="${JOB_NAME}" \
        --output_dir="./outputs/train/${JOB_NAME}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --dataset.image_transforms.enable=true \
        --dataset.image_transforms.max_num_transforms=3 \
        --dataset.image_transforms.random_order=false \
        --policy.type=dit \
        --policy.pretrained_path="${PRETRAINED_PATH}" \
        --policy.push_to_hub=${PUSH_TO_HUB} \
        --policy.repo_id="${REPO_ID}" \
        --policy.optimizer_lr=0.0002 \
        --batch_size=${BATCH_SIZE} \
        --num_workers=16 \
        --steps=${STEPS} \
        --seed=${SEED} \
        --eval_freq=0 \
        --save_freq=${SAVE_FREQ} \
        --log_freq=${LOG_FREQ} \
        --wandb.enable=${WANDB_ENABLE} \
        --wandb.disable_artifact=true \
        --wandb.project=${WANDB_PROJECT} \
        --wandb.entity=470620104-technical-university-of-munich
done

echo "All LoRA DiT single-task finetuning runs completed!"
