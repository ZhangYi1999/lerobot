#!/bin/bash
# EWC continual learning on LIBERO-10 (10 tasks sequential)
# Each task: train with Fisher penalty → compute Fisher → save EWC state
#
# Usage: bash bash_scripts/run_ewc.sh

source "$(dirname "$0")/common.sh"

METHOD="ewc"
OUTPUT_DIR="${OUTPUT_BASE}/${METHOD}"

EWC_LAMBDA=50000.0
EWC_GAMMA=0.9
EWC_FISHER_BATCHES=200

PREV_CHECKPOINT=""
PREV_EWC_STATE=""

for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do
    log_task_start "$METHOD" "$TASK_ID"

    COMMON_ARGS=$(common_train_args "$TASK_ID")
    WANDB=$(wandb_args "$METHOD" "$TASK_ID")
    TASK_OUT="${OUTPUT_DIR}/task${TASK_ID}"
    EWC_STATE_OUT="${OUTPUT_DIR}/ewc_state_task${TASK_ID}.pt"

    # ── Build EWC-specific args ──────────────────────────────────────────────
    EWC_ARGS="--ewc_lambda=${EWC_LAMBDA} --ewc_gamma=${EWC_GAMMA} --ewc_fisher_batches=${EWC_FISHER_BATCHES}"
    EWC_ARGS="${EWC_ARGS} --ewc_save_path=${EWC_STATE_OUT}"

    if [ "$TASK_ID" -eq 0 ]; then
        # First task: no prior EWC state
        python -m lerobot.scripts.clare.ewc \
            ${COMMON_ARGS} \
            --policy.path=${BASE_MODEL} \
            --steps=${STEPS} \
            --batch_size=${BATCH_SIZE} \
            ${EWC_ARGS} \
            --output_dir="${TASK_OUT}" \
            --job_name="${METHOD}_task${TASK_ID}" \
            ${WANDB}
    else
        # Subsequent tasks: load previous checkpoint + EWC state
        python -m lerobot.scripts.clare.ewc \
            ${COMMON_ARGS} \
            --policy.path=${PREV_CHECKPOINT} \
            --steps=${STEPS} \
            --batch_size=${BATCH_SIZE} \
            ${EWC_ARGS} \
            --ewc_state_path=${PREV_EWC_STATE} \
            --output_dir="${TASK_OUT}" \
            --job_name="${METHOD}_task${TASK_ID}" \
            ${WANDB}
    fi

    PREV_CHECKPOINT="${TASK_OUT}/checkpoints/last"
    PREV_EWC_STATE="${EWC_STATE_OUT}"
    echo "Task ${TASK_ID} complete. Checkpoint: ${PREV_CHECKPOINT}"
    echo ""
done

echo "All ${NUM_TASKS} tasks complete for ${METHOD}."
