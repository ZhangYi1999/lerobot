#!/bin/bash
set -e

# ==============================================================================
# Merge per-task discriminator checkpoints into one, then eval on all datasets.
# ==============================================================================

SEED=1000
PRETRAINED_PATH="continuallearning/dit_fft_pretraining_v2_lerobot30_seed1000"
MERGED_OUTPUT="outputs/lora_to_clare/clare_converted_v2_merged/adapter"
CLARE_CFG="configs/peft/clare_dit"

export REUSE_PRETRAINED_NORMALIZATION=true
export NORM_STATS_FILE="configs/union_stats.json"

WANDB_PROJECT="clare_rebuttal"
WANDB_ENTITY="470620104-technical-university-of-munich"

# ===== Step 1: Merge =====
echo "=========================================="
echo "Step 1: Merging discriminator checkpoints"
echo "=========================================="

python src/lerobot/scripts/clare/merge_discriminators.py \
    --output_dir "${MERGED_OUTPUT}" \
    --task_checkpoints \
        outputs/train/dit_posttrainv2_clare_discriminator_dit_real_0_put_bowl_filtered_seed1000/checkpoints/last/adapter \
        outputs/train/dit_posttrainv2_clare_discriminator_dit_real_1_stack_bowls_filtered_seed1000/checkpoints/last/adapter \
        outputs/train/dit_posttrainv2_clare_discriminator_dit_real_2_put_moka_pot_filtered_seed1000/checkpoints/last/adapter \
        outputs/train/dit_posttrainv2_clare_discriminator_dit_real_3_close_drawer_filtered_seed1000/checkpoints/last/adapter \
        outputs/train/dit_posttrainv2_clare_discriminator_dit_real_4_put_lego_into_drawer_filtered_seed1000/checkpoints/last/adapter

DATASETS=(
    "real_0_put_bowl_filtered"
    "real_1_stack_bowls_filtered"
    "real_2_put_moka_pot_filtered"
    "real_3_close_drawer_filtered"
    "real_4_put_lego_into_drawer_filtered"
)

# ===== Step 2: Eval on each dataset =====
echo ""
echo "=========================================="
echo "Step 2: Evaluating merged checkpoint"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    OUTPUT_DIR="outputs/eval_on_dataset/clare_merged_${DATASET}_seed${SEED}"
    JOB_NAME="eval_clare_merged_${DATASET}_s${SEED}"

    echo ""
    echo "--- Eval on ${DATASET} ---"

    PEFT_CONFIG_PATH="${MERGED_OUTPUT}" \
    PEFT_WEIGHT_PATH="${MERGED_OUTPUT}" \
    python -m lerobot.scripts.lerobot_eval_on_dataset \
        --policy.path="${PRETRAINED_PATH}" \
        --dataset.repo_id="continuallearning/${DATASET}" \
        --output_dir="${OUTPUT_DIR}" \
        --seed=${SEED} \
        --job_name="${JOB_NAME}" \
        --wandb.enable=true \
        --wandb.disable_artifact=true \
        --wandb.project=${WANDB_PROJECT} \
        --wandb.entity=${WANDB_ENTITY}

    echo "  Done. Results: ${OUTPUT_DIR}/metrics.json"
done

echo ""
echo "=========================================="
echo "All merge + eval complete!"
echo "=========================================="
