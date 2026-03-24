#!/bin/bash
set -e

# ==============================================================================
# Stage 2: Convert LoRA checkpoints to CLARE format
# Takes N LoRA adapter dirs and produces a single CLARE checkpoint with
# adapters populated and discriminators randomly initialized.
# ==============================================================================

# ===== Configuration =====
PRETRAINED_PATH="continuallearning/dit_fft_pretraining_v2_lerobot30_seed1000"
CLARE_CONFIG_PATH="configs/peft/clare_dit"
OUTPUT_DIR="outputs/lora_to_clare/clare_converted_v2"
export REUSE_PRETRAINED_NORMALIZATION="${REUSE_PRETRAINED_NORMALIZATION:-true}"

LORA_REPOS=(
    "continuallearning/dit_posttrainv2_baseline_lora_dit_all_real_0_put_bowl_filtered_seed1000"
    "continuallearning/dit_posttrainv2_baseline_lora_dit_all_real_1_stack_bowls_filtered_seed1000"
    "continuallearning/dit_posttrainv2_baseline_lora_dit_all_real_2_put_moka_pot_filtered_seed1000"
    "continuallearning/dit_posttrainv2_baseline_lora_dit_all_real_3_close_drawer_filtered_seed1000"
    "continuallearning/dit_posttrainv2_baseline_lora_dit_all_real_4_put_lego_into_drawer_filtered_seed1000"
)

N_TASKS=${#LORA_REPOS[@]}

echo "=========================================="
echo "Convert LoRA → CLARE"
echo "  Tasks:        ${N_TASKS}"
echo "  Pretrained:   ${PRETRAINED_PATH}"
echo "  CLARE config: ${CLARE_CONFIG_PATH}"
echo "  Output:       ${OUTPUT_DIR}"
echo "=========================================="

LORA_JSON=$(printf '%s' '['; sep=''; for r in "${LORA_REPOS[@]}"; do printf '%s"%s"' "$sep" "$r"; sep=','; done; printf ']')

python -m lerobot.scripts.clare.convert_lora_to_clare \
    --lora_checkpoint_dirs="${LORA_JSON}" \
    --clare_config_path="${CLARE_CONFIG_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --policy.type=dit \
    --policy.pretrained_path="${PRETRAINED_PATH}" \
    --policy.push_to_hub=false \
    --dataset.repo_id="continuallearning/real_0_put_bowl_filtered" \
    --eval_freq=0

echo ""
echo "Conversion complete: ${OUTPUT_DIR}"
echo "Next: run clare_finetune_dit.sh to train discriminators"
