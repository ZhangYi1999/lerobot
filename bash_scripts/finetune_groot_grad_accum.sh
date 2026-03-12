#!/bin/bash
set -e

BATCH_SIZE=8
NUM_WORKERS=8
STEPS=10000
SAVE_FREQ=10000
LOG_FREQ=100
GRAD_ACCUM_STEPS=4

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
    echo "Finetuning GROOT (grad accum ${GRAD_ACCUM_STEPS}x) on: ${DATASET}"
    echo "=========================================="
    python -m lerobot.scripts.lerobot_train_gradient_accumulation \
        --job_name="groot_fft_${STEPS}steps_ga${GRAD_ACCUM_STEPS}_${DATASET}" \
        --output_dir="./outputs/train/groot_fft_${STEPS}steps_ga${GRAD_ACCUM_STEPS}_${DATASET}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --policy.type=groot \
        --policy.push_to_hub=true \
        --policy.repo_id="continuallearning/groot_fft_${STEPS}steps_ga${GRAD_ACCUM_STEPS}_${DATASET}" \
        --gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
        --batch_size=${BATCH_SIZE} \
        --num_workers=${NUM_WORKERS} \
        --steps=${STEPS} \
        --eval_freq=0 \
        --save_freq=${SAVE_FREQ} \
        --log_freq=${LOG_FREQ} \
        --wandb.enable=true \
        --wandb.disable_artifact=true \
        --wandb.project=clare_rebuttal \
        --wandb.entity=470620104-technical-university-of-munich
done

echo "All GROOT finetuning (grad accum) runs completed!"
