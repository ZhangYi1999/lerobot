#!/bin/bash
set -e

STEPS=40000
SAVE_FREQ=40000

DATASETS=(
    "real_0_put_bowl_filtered"
    "real_1_stack_bowls_filtered"
    "real_0_put_bowl"
    "real_1_stack_bowls"
    # "real_2_put_moka_pot"
    # "real_2_put_moka_pot_filtered"
    # "real_3_close_drawer"
    # "real_3_close_drawer_filtered"
    # "real_4_put_lego_into_drawer"
    # "real_4_put_lego_into_drawer_filtered"
    # "real_5_stack_lego"
    # "real_5_stack_lego_filtered"
)

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Training diffusion on: ${DATASET}"
    echo "=========================================="
    lerobot-train \
        --job_name="diffusion_fft_${DATASET}" \
        --output_dir="./outputs/train/diffusion_fft_${DATASET}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --policy.type=diffusion \
        --policy.push_to_hub=true \
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
