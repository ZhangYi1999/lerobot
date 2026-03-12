#!/bin/bash
set -e

BATCH_SIZE=8
NUM_WORKERS=8
STEPS=200
SAVE_FREQ=200

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
    echo "Finetuning GROOT on: ${DATASET}"
    echo "=========================================="
    lerobot-train \
        --job_name="groot_fft_${STEPS}steps_${DATASET}" \
        --output_dir="./outputs/train/groot_fft_${STEPS}steps_${DATASET}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --policy.type=groot \
        --policy.push_to_hub=true \
        --policy.repo_id="continuallearning/groot_fft_${STEPS}steps_${DATASET}" \
        --batch_size=${BATCH_SIZE} \
        --num_workers=${NUM_WORKERS} \
        --steps=${STEPS} \
        --eval_freq=0 \
        --save_freq=${SAVE_FREQ} \
        --log_freq=1 \
        --wandb.enable=true \
        --wandb.disable_artifact=true \
        --wandb.project=clare_rebuttal \
        --wandb.entity=470620104-technical-university-of-munich
done

echo "All GROOT finetuning runs completed!"
