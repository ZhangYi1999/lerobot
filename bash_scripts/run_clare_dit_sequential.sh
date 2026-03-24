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

    # PEFT env vars: task 0 uses config, subsequent tasks load previous checkpoint
    if [ $i -eq 0 ]; then
        export PEFT_CFG_PATH="${PEFT_CFG}"
        unset PEFT_WEIGHT_PATH
    else
        export PEFT_WEIGHT_PATH="${PREV_CHECKPOINT}/checkpoints/last/adapter/default"
        unset PEFT_CFG_PATH
    fi

    # CLARE-specific env vars
    export CLARE_PHASE="full"
    export TRAIN_DISCRIMINATORS_STEPS=${DISC_STEPS}
    export EXPAND_THRESHOLD=0.0
    export AT_LEAST_EXPAND=shallowest

    echo "=========================================="
    echo "CLARE Task ${i}: ${DATASET}"
    echo "=========================================="

    accelerate launch --mixed_precision=${MIXED_PRECISION} \
        -m lerobot.scripts.clare.clare \
        --use_policy_training_preset=false \
        --policy.type=dit \
        --policy.pretrained_path="${PRETRAINED_PATH}" \
        --policy.push_to_hub=false \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --batch_size=${BATCH_SIZE} \
        --steps=${STEPS} \
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
