#!/bin/bash
set -e

# ===== Configuration =====
NUM_BLOCKS=12
STEPS=10000
SAVE_FREQ=5000
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"  # Options: no, fp16, bf16

SEEDS=(1000)

DATASETS=(
    "real_0_put_bowl_filtered"
    # "real_1_stack_bowls_filtered"
    # "real_2_put_moka_pot_filtered"
    # "real_3_close_drawer_filtered"
)

# ===== Training Loop =====
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "=========================================="
        echo "Training larger DiT (num_blocks=${NUM_BLOCKS}) on: ${DATASET} (seed=${SEED})"
        echo "=========================================="
        accelerate launch --mixed_precision=${MIXED_PRECISION} \
            -m lerobot.scripts.lerobot_train \
            --job_name="dit_larger_fft_${DATASET}_seed${SEED}" \
            --output_dir="./outputs/train/dit_larger_fft_${DATASET}_seed${SEED}" \
            --dataset.repo_id="continuallearning/${DATASET}" \
            --policy.type=dit \
            --policy.num_blocks=${NUM_BLOCKS} \
            --policy.push_to_hub=true \
            --policy.repo_id="continuallearning/dit_larger_fft_${DATASET}_seed${SEED}" \
            --batch_size=128 \
            --num_workers=16 \
            --steps=${STEPS} \
            --seed=${SEED} \
            --eval_freq=0 \
            --save_freq=${SAVE_FREQ} \
            --log_freq=100 \
            --wandb.enable=true \
            --wandb.disable_artifact=true \
            --wandb.project=clare_rebuttal \
            --wandb.entity=470620104-technical-university-of-munich
    done
done

echo "All larger DiT training runs completed!"
