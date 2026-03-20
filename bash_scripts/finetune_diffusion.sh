#!/bin/bash
set -e

STEPS=100
SAVE_FREQ=100
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"  # Options: no, fp16, bf16

DATASETS=(
    # "real_0_put_bowl_filtered"
    # "real_1_stack_bowls_filtered"
    # "real_0_put_bowl"
    # "real_1_stack_bowls"
    # "real_2_put_moka_pot_filtered"
    # "real_3_close_drawer_filtered"
    # "real_3_close_drawer"
    # "real_2_put_moka_pot"
    # "real_4_put_lego_into_drawer"
    # "real_5_stack_lego"
    # "real_5_stack_lego_filtered"
    # "real_2_put_moka_pot_filtered_fixed"
    # "real_3_close_drawer_filtered_fixed"
    # "real_3_close_drawer_fixed"
    # "real_2_put_moka_pot_fixed"
    # "real_4_put_lego_into_drawer"
    "real_4_put_lego_into_drawer_filtered"
)

# HF_LEROBOT_HOME defaults to ${HF_HOME}/lerobot (or ~/.cache/huggingface/lerobot)
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot}"

clear_dataset_cache() {
    local dataset=$1
    local cache_path="${HF_LEROBOT_HOME}/continuallearning/${dataset}"
    if [ -d "${cache_path}" ]; then
        echo "Clearing stale cache: ${cache_path}"
        rm -rf "${cache_path}"
    fi
}

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Training diffusion on: ${DATASET}"
    echo "=========================================="
    accelerate launch --mixed_precision=${MIXED_PRECISION} \
        -m lerobot.scripts.lerobot_train \
        --job_name="diffusion_fft_${DATASET}" \
        --output_dir="./outputs/train/diffusion_fft_${DATASET}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --policy.type=diffusion \
        --policy.push_to_hub=false \
        --policy.repo_id="continuallearning/diffusion_fft_${DATASET}" \
        --batch_size=64 \
        --num_workers=16 \
        --steps=${STEPS} \
        --eval_freq=0 \
        --save_freq=${SAVE_FREQ} \
        --log_freq=100 \
        --wandb.enable=true \
        --wandb.disable_artifact=true \
        --wandb.project=clare_rebuttal \
        --wandb.entity=470620104-technical-university-of-munich
done

echo "All training runs completed!"
