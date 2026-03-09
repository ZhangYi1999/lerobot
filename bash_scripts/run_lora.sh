#!/bin/bash
# SeqLoRA continual learning on LIBERO-10 (10 tasks sequential)
# Each task: add LoRA → train → merge into backbone → next task
#
# Usage: bash bash_scripts/run_lora.sh [PEFT_CFG_PATH]
#   PEFT_CFG_PATH: path to LoRA adapter_config.json directory (default: configs/peft/lora)

source "$(dirname "$0")/common.sh"

METHOD="lora"
OUTPUT_DIR="${OUTPUT_BASE}/${METHOD}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PEFT_CFG_PATH="${1:-${SCRIPT_DIR}/../configs/peft/lora}"

PREV_CHECKPOINT=""

for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do
    log_task_start "$METHOD" "$TASK_ID"

    COMMON_ARGS=$(common_train_args "$TASK_ID")
    WANDB=$(wandb_args "$METHOD" "$TASK_ID")
    TASK_OUT="${OUTPUT_DIR}/task${TASK_ID}"

    if [ "$TASK_ID" -eq 0 ]; then
        # First task: init from base model + LoRA config
        python -m lerobot.scripts.clare.lora \
            ${COMMON_ARGS} \
            --policy.path=${BASE_MODEL} \
            --peft_cfg_path=${PEFT_CFG_PATH} \
            --merge_back_to_policy=true \
            --steps=${STEPS} \
            --batch_size=${BATCH_SIZE} \
            --output_dir="${TASK_OUT}" \
            --job_name="${METHOD}_task${TASK_ID}" \
            ${WANDB}
    else
        # Subsequent tasks: load merged checkpoint, apply fresh LoRA
        python -m lerobot.scripts.clare.lora \
            ${COMMON_ARGS} \
            --policy.path=${PREV_CHECKPOINT} \
            --peft_cfg_path=${PEFT_CFG_PATH} \
            --merge_back_to_policy=true \
            --steps=${STEPS} \
            --batch_size=${BATCH_SIZE} \
            --output_dir="${TASK_OUT}" \
            --job_name="${METHOD}_task${TASK_ID}" \
            ${WANDB}
    fi

    PREV_CHECKPOINT="${TASK_OUT}/checkpoints/last"
    echo "Task ${TASK_ID} complete. Checkpoint: ${PREV_CHECKPOINT}"
    echo ""
done

echo "All ${NUM_TASKS} tasks complete for ${METHOD}."
