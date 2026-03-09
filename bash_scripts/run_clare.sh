#!/bin/bash
# CLARE continual learning on LIBERO-10 (10 tasks sequential)
# Each task: Phase 1 (adapter) → Phase 2 (discriminator)
#
# Usage: bash bash_scripts/run_clare.sh [PEFT_CFG_PATH]
#   PEFT_CFG_PATH: path to CLARE adapter_config.json directory (default: configs/peft/clare)

source "$(dirname "$0")/common.sh"

METHOD="clare"
OUTPUT_DIR="${OUTPUT_BASE}/${METHOD}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PEFT_CFG_PATH="${1:-${SCRIPT_DIR}/../configs/peft/clare}"

PREV_CHECKPOINT=""

for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do
    log_task_start "$METHOD" "$TASK_ID"

    COMMON_ARGS=$(common_train_args "$TASK_ID")
    WANDB=$(wandb_args "$METHOD" "$TASK_ID")

    ADAPTER_OUT="${OUTPUT_DIR}/task${TASK_ID}_adapter"
    DISC_OUT="${OUTPUT_DIR}/task${TASK_ID}_disc"

    # ── Determine PEFT loading args ──────────────────────────────────────────
    if [ "$TASK_ID" -eq 0 ]; then
        # First task: initialize from config
        PEFT_ARGS="--peft_cfg_path=${PEFT_CFG_PATH}"
        POLICY_ARGS="--policy.path=${BASE_MODEL}"
    else
        # Subsequent tasks: load previous discriminator checkpoint's adapter weights
        PEFT_ARGS="--peft_weight_path=${PREV_CHECKPOINT}/adapter"
        POLICY_ARGS="--policy.path=${PREV_CHECKPOINT}"
    fi

    # ── Phase 1: Train adapters ──────────────────────────────────────────────
    echo "[Phase 1] Adapter training — Task ${TASK_ID}"
    python -m lerobot.scripts.clare.clare \
        ${COMMON_ARGS} \
        ${POLICY_ARGS} \
        ${PEFT_ARGS} \
        --phase=adapter \
        --steps=${STEPS} \
        --batch_size=${BATCH_SIZE} \
        --output_dir="${ADAPTER_OUT}" \
        --job_name="${METHOD}_task${TASK_ID}_adapter" \
        ${WANDB}

    ADAPTER_CKPT="${ADAPTER_OUT}/checkpoints/last/adapter"

    # ── Phase 2: Train discriminators ────────────────────────────────────────
    echo "[Phase 2] Discriminator training — Task ${TASK_ID}"
    python -m lerobot.scripts.clare.clare \
        ${COMMON_ARGS} \
        ${POLICY_ARGS} \
        ${PEFT_ARGS} \
        --phase=discriminator \
        --adapter_checkpoint_path="${ADAPTER_CKPT}" \
        --output_dir="${DISC_OUT}" \
        --job_name="${METHOD}_task${TASK_ID}_disc" \
        ${WANDB}

    # Update checkpoint pointer for next task
    PREV_CHECKPOINT="${DISC_OUT}/checkpoints/last"

    echo "Task ${TASK_ID} complete. Checkpoint: ${PREV_CHECKPOINT}"
    echo ""
done

echo "All ${NUM_TASKS} tasks complete for ${METHOD}."
