#!/bin/bash
# Experience Replay continual learning on LIBERO-10 (10 tasks sequential)
# Task 0: normal training. Task N≥1: train on current task + replay from all previous.
#
# Usage: bash bash_scripts/run_er.sh

source "$(dirname "$0")/common.sh"

METHOD="er"
OUTPUT_DIR="${OUTPUT_BASE}/${METHOD}"

# ER uses split batch: 16 current + 16 replay = 32 total
ER_BATCH_SIZE=16
REPLAY_BATCH_SIZE=16

PREV_CHECKPOINT=""

for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do
    log_task_start "$METHOD" "$TASK_ID"

    COMMON_ARGS=$(common_train_args "$TASK_ID")
    WANDB=$(wandb_args "$METHOD" "$TASK_ID")
    TASK_OUT="${OUTPUT_DIR}/task${TASK_ID}"

    if [ "$TASK_ID" -eq 0 ]; then
        # ── First task: no replay buffer ─────────────────────────────────────
        python -m lerobot.scripts.clare.er \
            ${COMMON_ARGS} \
            --policy.path=${BASE_MODEL} \
            --steps=${STEPS} \
            --batch_size=${ER_BATCH_SIZE} \
            --output_dir="${TASK_OUT}" \
            --job_name="${METHOD}_task${TASK_ID}" \
            ${WANDB}
    else
        # ── Subsequent tasks: current + replay ───────────────────────────────
        # Replay episodes = all episodes from task 0..task_id-1
        REPLAY_EPISODES=$(get_cumulative_episodes $((TASK_ID - 1)))

        python -m lerobot.scripts.clare.er \
            ${COMMON_ARGS} \
            --policy.path=${PREV_CHECKPOINT} \
            --steps=${STEPS} \
            --batch_size=${ER_BATCH_SIZE} \
            --replay_dataset.repo_id=${DATASET_REPO} \
            --replay_dataset.episodes="${REPLAY_EPISODES}" \
            --replay_batch_size=${REPLAY_BATCH_SIZE} \
            --output_dir="${TASK_OUT}" \
            --job_name="${METHOD}_task${TASK_ID}" \
            ${WANDB}
    fi

    PREV_CHECKPOINT="${TASK_OUT}/checkpoints/last"
    echo "Task ${TASK_ID} complete. Checkpoint: ${PREV_CHECKPOINT}"
    echo ""
done

echo "All ${NUM_TASKS} tasks complete for ${METHOD}."
