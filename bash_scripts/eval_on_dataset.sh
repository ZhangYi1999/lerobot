#!/bin/bash
set -e

# ==============================================================================
# Evaluate trained checkpoints on dataset (predicted vs ground truth actions).
# Supports two modes:
#   lora  — LoRA adapter on top of a shared pretrained base model
#   fft   — fully fine-tuned checkpoint (no PEFT)
#
# Run format (pipe-separated):
#   lora|<lora_config>|<dataset>|<checkpoint_dir>
#   fft |<label>      |<dataset>|<checkpoint_dir>
#
# Usage:
#   bash bash_scripts/eval_on_dataset.sh
# ==============================================================================

# ===== Configuration =====
PYTHON="${PYTHON:-/home/yi/miniconda3/envs/lerobot/bin/python}"
SEED=1000
# Base pretrained model used for LoRA runs (not used for fft runs)
LORA_BASE_MODEL="outputs/train/dit_fft_pretraining_v1_s1000/checkpoints/last/pretrained_model"

# WandB settings
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="clare_rebuttal"
WANDB_ENTITY="470620104-technical-university-of-munich"

# ===== Resolve paths =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# ===== Evaluation runs: TYPE | CONFIG_LABEL | DATASET | CHECKPOINT_DIR =====
RUNS=(
    # LoRA runs
    "lora|dit_all|real_0_put_bowl_filtered|outputs/train/dit_lora_dit_all_real_0_put_bowl_filtered_seed1000"
    "lora|dit_all|real_1_stack_bowls_filtered|outputs/train/dit_lora_dit_all_real_1_stack_bowls_filtered_seed1000"
    "lora|dit_encoder|real_0_put_bowl_filtered|outputs/train/dit_lora_dit_encoder_real_0_put_bowl_filtered_seed1000"
    "lora|dit_encoder|real_1_stack_bowls_filtered|outputs/train/dit_lora_dit_encoder_real_1_stack_bowls_filtered_seed1000"
    # FFT runs
    "fft|diffusion_from_scratch|real_0_put_bowl_filtered|outputs/train/diffusion_fft_real_0_put_bowl_filtered"
    "fft|dit_from_scratch|real_0_put_bowl_filtered|outputs/train/dit_fft_real_0_put_bowl_filtered_seed1000"
    "fft|dit_larger_from_scratch|real_0_put_bowl_filtered|outputs/train/dit_larger_fft_real_0_put_bowl_filtered_seed1000"
)

echo "=========================================="
echo "Dataset Evaluation"
echo "  Runs: ${#RUNS[@]}"
echo "=========================================="

for RUN in "${RUNS[@]}"; do
    IFS='|' read -r TYPE CONFIG_LABEL DATASET CKPT_DIR <<< "${RUN}"

    echo ""
    echo "--- ${TYPE} / ${CONFIG_LABEL} / ${DATASET} ---"

    CKPT_MODEL="${CKPT_DIR}/checkpoints/last/pretrained_model"
    if [ ! -d "${CKPT_MODEL}" ]; then
        echo "SKIP: Checkpoint dir not found at ${CKPT_MODEL}"
        continue
    fi

    OUTPUT_DIR="outputs/eval_on_dataset/${TYPE}_${CONFIG_LABEL}_${DATASET}_seed${SEED}"
    JOB_NAME="eval_${TYPE}_${CONFIG_LABEL}_${DATASET}_s${SEED}"

    if [ "${TYPE}" = "lora" ]; then
        LORA_CFG_PATH="${REPO_ROOT}/configs/lora/${CONFIG_LABEL}"
        if [ ! -f "${LORA_CFG_PATH}/adapter_config.json" ]; then
            echo "ERROR: LoRA config not found at ${LORA_CFG_PATH}"
            exit 1
        fi
        echo "  Type:         lora"
        echo "  LoRA config:  ${LORA_CFG_PATH}"
        echo "  Base model:   ${LORA_BASE_MODEL}"
        echo "  PEFT weights: ${CKPT_MODEL}"
        echo "  Output dir:   ${OUTPUT_DIR}"

        PEFT_CONFIG_PATH="${LORA_CFG_PATH}" \
        PEFT_WEIGHT_PATH="${CKPT_MODEL}" \
        ${PYTHON} -m lerobot.scripts.lerobot_eval_on_dataset \
            --policy.path="${LORA_BASE_MODEL}" \
            --dataset.repo_id="continuallearning/${DATASET}" \
            --output_dir="${OUTPUT_DIR}" \
            --seed=${SEED} \
            --job_name="${JOB_NAME}" \
            --wandb.enable=${WANDB_ENABLE} \
            --wandb.project=${WANDB_PROJECT} \
            --wandb.entity=${WANDB_ENTITY}

    elif [ "${TYPE}" = "fft" ]; then
        echo "  Type:         fft"
        echo "  Checkpoint:   ${CKPT_MODEL}"
        echo "  Output dir:   ${OUTPUT_DIR}"

        ${PYTHON} -m lerobot.scripts.lerobot_eval_on_dataset \
            --policy.path="${CKPT_MODEL}" \
            --dataset.repo_id="continuallearning/${DATASET}" \
            --output_dir="${OUTPUT_DIR}" \
            --seed=${SEED} \
            --job_name="${JOB_NAME}" \
            --wandb.enable=${WANDB_ENABLE} \
            --wandb.project=${WANDB_PROJECT} \
            --wandb.entity=${WANDB_ENTITY}

    else
        echo "ERROR: Unknown TYPE '${TYPE}'. Expected 'lora' or 'fft'."
        exit 1
    fi

    echo "  Done. Results: ${OUTPUT_DIR}/metrics.json"
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
