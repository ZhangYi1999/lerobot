#!/bin/bash
set -e

PRETRAINED_PATH="continuallearning/dit_fft_pretraining_v2_lerobot30_seed1000"
PEFT_CFG="configs/peft/clare_dit"
SEED=1000
STEPS=40000
DISC_STEPS=10000
BATCH_SIZE=128
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export REUSE_PRETRAINED_NORMALIZATION="${REUSE_PRETRAINED_NORMALIZATION:-true}"
OUTPUT_BASE="outputs/clare_dit"

DATASETS=(
    "real_0_put_bowl_filtered"
    "real_1_stack_bowls_filtered"
    "real_2_put_moka_pot_filtered"
    "real_3_close_drawer_filtered"
)
TASK_NAMES=(
    "task0_put_bowl"
    "task1_stack_bowls"
    "task2_put_moka_pot"
    "task3_close_drawer"
)

PREV_CHECKPOINT=""

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    TASK_NAME="${TASK_NAMES[$i]}"
    OUTPUT_DIR="${OUTPUT_BASE}/${TASK_NAME}"

    # Build PEFT args: task 0 uses config, subsequent tasks load previous checkpoint
    if [ $i -eq 0 ]; then
        PEFT_ARGS="--peft_cfg_path=${PEFT_CFG}"
    else
        PEFT_ARGS="--peft_weight_path=${PREV_CHECKPOINT}/checkpoints/last/adapter/default"
    fi

    echo "=========================================="
    echo "CLARE Task ${i}: ${DATASET}"
    echo "=========================================="

    accelerate launch --mixed_precision=${MIXED_PRECISION} \
        -m lerobot.scripts.clare.clare \
        --phase=full \
        --policy.type=dit \
        --policy.pretrained_path="${PRETRAINED_PATH}" \
        --policy.push_to_hub=false \
        ${PEFT_ARGS} \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --batch_size=${BATCH_SIZE} \
        --steps=${STEPS} \
        --train_discriminators_steps=${DISC_STEPS} \
        --expand_threshold=0.0 \
        --at_least_expand=shallowest \
        --seed=${SEED} \
        --num_workers=16 \
        --eval_freq=0 \
        --save_freq=${STEPS} \
        --log_freq=100 \
        --output_dir="${OUTPUT_DIR}" \
        --wandb.enable=true \
        --wandb.disable_artifact=true \
        --wandb.project=clare_rebuttal \
        --wandb.entity=470620104-technical-university-of-munich \
        --job_name="clare_dit_${TASK_NAME}_s${SEED}"

    PREV_CHECKPOINT="${OUTPUT_DIR}"
done

echo "All CLARE sequential tasks completed!"
