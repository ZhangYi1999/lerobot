#!/bin/bash
set -e

# ===== Configuration =====
STEPS=20000
SAVE_FREQ=20000
LOG_FREQ=100
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
START_TASK="${START_TASK:-1}"   # Set to resume from middle, e.g. START_TASK=1
SEED=1000

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
    "continuallearning/dit_posttrainv2_real_0_put_bowl_filtered_seed1000"
    "continuallearning/dit_posttrainv2_seqfft_real_1_stack_bowls_filtered_seed1000"
    "continuallearning/dit_posttrainv2_seqfft_real_2_put_moka_pot_filtered_seed1000"
    "continuallearning/dit_posttrainv2_seqfft_real_3_close_drawer_filtered_seed1000"
    # "outputs/train/dit_posttrainv2_seqfft_real_1_stack_bowls_filtered_seed1000/checkpoints/last/pretrained_model"
    # "outputs/train/dit_posttrainv2_seqfft_real_2_put_moka_pot_filtered_seed1000/checkpoints/last/pretrained_model"
)

# Get output hub repo_id for a given task index and dataset name
get_repo_id() {
    local idx=$1
    local dataset=$2
    if [ "$idx" -eq 0 ]; then
        echo "continuallearning/dit_posttrainv2_${dataset}_seed${SEED}"
    else
        echo "continuallearning/dit_posttrainv2_seqfft_${dataset}_seed${SEED}"
    fi
}

# ===== Training Loop =====
for i in "${!DATASETS[@]}"; do
    if [ "$i" -lt "$START_TASK" ]; then
        echo "Skipping task ${i} (START_TASK=${START_TASK})"
        continue
    fi

    DATASET="${DATASETS[$i]}"
    REPO_ID=$(get_repo_id $i "${DATASET}")
    CURRENT_PRETRAINED="${CHECKPOINTS[$i]}"

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
        --policy.type=dit \
        --policy.pretrained_path="${CURRENT_PRETRAINED}" \
        --policy.push_to_hub=true \
        --policy.repo_id="${REPO_ID}" \
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
        --wandb.entity=470620104-technical-university-of-munich
done

echo "All SeqFFT DiT posttrain runs completed!"
