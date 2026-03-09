#!/bin/bash
# Common configuration for CLARE continual learning experiments on LIBERO-10
# Source this file from individual method scripts: source "$(dirname "$0")/common.sh"

set -euo pipefail

# ─── Dataset ────────────────────────────────────────────────────────────────────
DATASET_REPO="lerobot/libero_10_subtask"
NUM_TASKS=10
EPISODES_PER_TASK=50

# Episode-to-task mapping (to be confirmed by verify_dataset.sh)
# Assumes task i = episodes [i*50 .. i*50+49]
get_episode_list() {
    local task_id=$1
    local start=$((task_id * EPISODES_PER_TASK))
    local end=$((start + EPISODES_PER_TASK - 1))
    # Build JSON list: [start, start+1, ..., end]
    local list="["
    for i in $(seq $start $end); do
        if [ $i -gt $start ]; then list="${list},"; fi
        list="${list}${i}"
    done
    list="${list}]"
    echo "$list"
}

# Build seen task_ids list [0,1,...,task_id] for evaluation
get_seen_task_ids() {
    local task_id=$1
    local list="["
    for i in $(seq 0 $task_id); do
        if [ $i -gt 0 ]; then list="${list},"; fi
        list="${list}${i}"
    done
    list="${list}]"
    echo "$list"
}

# Build cumulative episode list for all tasks [0..task_id]
get_cumulative_episodes() {
    local task_id=$1
    local start=0
    local end=$(( (task_id + 1) * EPISODES_PER_TASK - 1 ))
    local list="["
    for i in $(seq $start $end); do
        if [ $i -gt 0 ]; then list="${list},"; fi
        list="${list}${i}"
    done
    list="${list}]"
    echo "$list"
}

# ─── Training defaults ──────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.5-3B}"
STEPS=10000
BATCH_SIZE=32
SEED=42
EVAL_N_EPISODES=10
NUM_WORKERS=16

# ─── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_BASE="outputs"

# ─── WandB ──────────────────────────────────────────────────────────────────────
WANDB_ENABLED=${WANDB_ENABLED:-true}
WANDB_PROJECT=${WANDB_PROJECT:-"clare-rebuttal"}

wandb_args() {
    local method=$1
    local task_id=$2
    if [ "$WANDB_ENABLED" = "true" ]; then
        echo "--wandb.enable=true --wandb.project=${WANDB_PROJECT} --wandb.run_name=${method}/task${task_id}"
    else
        echo "--wandb.enable=false"
    fi
}

# ─── Common training args ───────────────────────────────────────────────────────
common_train_args() {
    local task_id=$1
    local episodes
    episodes=$(get_episode_list "$task_id")
    local seen_tasks
    seen_tasks=$(get_seen_task_ids "$task_id")
    echo "--dataset.repo_id=${DATASET_REPO}" \
         "--dataset.episodes=${episodes}" \
         "--env.type=libero" \
         "--env.task=libero_10" \
         "--env.task_ids=${seen_tasks}" \
         "--seed=${SEED}" \
         "--num_workers=${NUM_WORKERS}" \
         "--eval.n_episodes=${EVAL_N_EPISODES}"
}

log_task_start() {
    local method=$1
    local task_id=$2
    echo "========================================"
    echo " ${method} — Task ${task_id} / $((NUM_TASKS - 1))"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
}
