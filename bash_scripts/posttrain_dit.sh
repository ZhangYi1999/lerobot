#!/bin/bash
set -e

# ===== Configuration =====
PRETRAINED_PATH="outputs/train/dit_fft_pretraining_v1_s1000/checkpoints/last/pretrained_model"
STEPS=10000
SAVE_FREQ=5000
LOG_FREQ=100
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"  # Options: no, fp16, bf16
export REUSE_PRETRAINED_NORMALIZATION="${REUSE_PRETRAINED_NORMALIZATION:-true}"

SEEDS=(1000)

DATASETS=(
    "real_0_put_bowl_filtered"
    "real_1_stack_bowls_filtered"
    "real_2_put_moka_pot_filtered"
    "real_3_close_drawer_filtered"
)

# ===== Training Loop =====
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "=========================================="
        echo "Posttrain DiT on: ${DATASET} (seed=${SEED})"
        echo "=========================================="
        accelerate launch --mixed_precision=${MIXED_PRECISION} \
            -m lerobot.scripts.lerobot_train \
            --job_name="dit_posttrain_fft_${DATASET}_seed${SEED}" \
            --output_dir="./outputs/train/dit_posttrain_fft_${DATASET}_seed${SEED}" \
            --dataset.repo_id="continuallearning/${DATASET}" \
            --policy.type=dit \
            --policy.pretrained_path="${PRETRAINED_PATH}" \
            --policy.push_to_hub=true \
            --policy.repo_id="continuallearning/dit_posttrain_${DATASET}_seed${SEED}" \
            --batch_size=128 \
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
done

echo "All DiT posttrain runs completed!"
