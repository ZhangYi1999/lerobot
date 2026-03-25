#!/bin/bash
set -e

# ===== Configuration =====
STEPS=20000
SAVE_FREQ=20000
LOG_FREQ=100
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export REUSE_PRETRAINED_NORMALIZATION="${REUSE_PRETRAINED_NORMALIZATION:-true}"
START_TASK="${START_TASK:-1}"   # Set to resume from middle, e.g. START_TASK=1
SEED=1000

# Normalization source control:
#   "pretrained" (default) — each task uses normalization from its own CHECKPOINTS[i]
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

# Pretrained checkpoint for each task (hub repo_id or local path).
# CHECKPOINTS[i] is the starting checkpoint for DATASETS[i].
CHECKPOINTS=(
    "continuallearning/dit_fft_pretraining_v2_lerobot30_seed1000"
    "continuallearning/dit_posttrainv2_seqfft_real_0_put_bowl_filtered_seed${SEED}"
    "continuallearning/dit_posttrainv2_seqfft_real_1_stack_bowls_filtered_seed${SEED}"
    "continuallearning/dit_posttrainv2_seqfft_real_2_put_moka_pot_filtered_seed${SEED}"
    "continuallearning/dit_posttrainv2_seqfft_real_3_close_drawer_filtered_seed${SEED}"
    # "outputs/train/dit_posttrainv2_seqfft_real_0_put_bowl_filtered_seed${SEED}/checkpoints/last/pretrained_model"
    # "outputs/train/dit_posttrainv2_seqfft_real_1_stack_bowls_filtered_seed${SEED}/checkpoints/last/pretrained_model"
    # "outputs/train/dit_posttrainv2_seqfft_real_2_put_moka_pot_filtered_seed${SEED}/checkpoints/last/pretrained_model"
    # "outputs/train/dit_posttrainv2_seqfft_real_3_close_drawer_filtered_seed${SEED}/checkpoints/last/pretrained_model"
)

# ===== Training Loop =====
for i in "${!DATASETS[@]}"; do
    if [ "$i" -lt "$START_TASK" ]; then
        echo "Skipping task ${i} (START_TASK=${START_TASK})"
        continue
    fi

    DATASET="${DATASETS[$i]}"
    REPO_ID="continuallearning/dit_posttrainv2_seqfft_${DATASET}_seed${SEED}"
    CURRENT_PRETRAINED="${CHECKPOINTS[$i]}"

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
            TASK0_REPO_ID="continuallearning/dit_posttrainv2_seqfft_${DATASETS[0]}_seed${SEED}"
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
    echo "SeqFFT DiT: ${DATASET} (task=${i}, seed=${SEED})"
    echo "  From: ${CURRENT_PRETRAINED}"
    echo "  To:   ${REPO_ID}"
    echo "=========================================="

    accelerate launch --mixed_precision=${MIXED_PRECISION} \
        -m lerobot.scripts.lerobot_train \
        --job_name="${JOB_NAME}" \
        --output_dir="./outputs/train/${JOB_NAME}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --dataset.image_transforms.enable=true \
        --policy.type=dit \
        --policy.pretrained_path="${CURRENT_PRETRAINED}" \
        --policy.push_to_hub=true \
        --policy.repo_id="${REPO_ID}" \
        --policy.freeze_language_proj=True \
        --policy.freeze_vision_proj=True \
        --policy.freeze_state_proj=True \
        --policy.optimizer_lr=0.0002 \
        --batch_size=256 \
        --num_workers=16 \
        --steps=${STEPS} \
        --seed=${SEED} \
        --eval_freq=0 \
        --save_freq=${SAVE_FREQ} \
        --log_freq=${LOG_FREQ} \
        --wandb.enable=true \
        --wandb.disable_artifact=true \
        --wandb.project=clare_rebuttal \
        --wandb.entity=470620104-technical-university-of-munich \
        --policy.freeze_language_proj=true \
        --policy.freeze_vision_proj=true \
        --policy.freeze_state_proj=true
done

echo "All SeqFFT DiT posttrain runs completed!"
