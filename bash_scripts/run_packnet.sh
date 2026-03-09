#!/bin/bash
# PackNet continual learning on LIBERO-10 (10 tasks sequential)
# Each task: train on free weights → prune 75% → post-prune fine-tune
#
# Usage: bash bash_scripts/run_packnet.sh

source "$(dirname "$0")/common.sh"

METHOD="packnet"
OUTPUT_DIR="${OUTPUT_BASE}/${METHOD}"

PRUNE_RATIO=0.75
POST_PRUNE_STEPS=20000

PREV_CHECKPOINT=""

for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do
    log_task_start "$METHOD" "$TASK_ID"

    COMMON_ARGS=$(common_train_args "$TASK_ID")
    WANDB=$(wandb_args "$METHOD" "$TASK_ID")
    TASK_OUT="${OUTPUT_DIR}/task${TASK_ID}"

    POLICY_ARG=""
    if [ "$TASK_ID" -eq 0 ]; then
        POLICY_ARG="--policy.path=${BASE_MODEL}"
    else
        POLICY_ARG="--policy.path=${PREV_CHECKPOINT}"
    fi

    python -m lerobot.scripts.clare.packnet \
        ${COMMON_ARGS} \
        ${POLICY_ARG} \
        --current_task=${TASK_ID} \
        --prune_ratio=${PRUNE_RATIO} \
        --steps=${STEPS} \
        --post_prune_steps=${POST_PRUNE_STEPS} \
        --batch_size=${BATCH_SIZE} \
        --output_dir="${TASK_OUT}" \
        --job_name="${METHOD}_task${TASK_ID}" \
        ${WANDB}

    PREV_CHECKPOINT="${TASK_OUT}/checkpoints/last"
    echo "Task ${TASK_ID} complete. Checkpoint: ${PREV_CHECKPOINT}"
    echo ""
done

echo "All ${NUM_TASKS} tasks complete for ${METHOD}."
