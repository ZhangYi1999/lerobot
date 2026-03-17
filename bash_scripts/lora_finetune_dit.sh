#!/bin/bash
set -e

# ===== Configuration =====
LORA_CONFIG="dit_all"  # Options: dit_encoder, dit_decoder, dit_all
STEPS=200000
SAVE_FREQ=20000
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"  # Options: no, fp16, bf16

SEEDS=(1000)

DATASETS=(
    # "real_0_put_bowl_filtered"
    # "real_1_stack_bowls_filtered"
    # "real_2_put_moka_pot_filtered"
    # "real_3_close_drawer_filtered"
)

# ===== Resolve PEFT config path =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
export PEFT_CONFIG_PATH="${REPO_ROOT}/configs/lora/${LORA_CONFIG}"

if [ ! -f "${PEFT_CONFIG_PATH}/adapter_config.json" ]; then
    echo "ERROR: adapter_config.json not found at ${PEFT_CONFIG_PATH}"
    exit 1
fi

echo "Using LoRA config: ${PEFT_CONFIG_PATH}/adapter_config.json"

# ===== Training Loop =====
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "=========================================="
        echo "LoRA finetuning DiT (${LORA_CONFIG}) on: ${DATASET} (seed=${SEED})"
        echo "=========================================="
        accelerate launch --mixed_precision=${MIXED_PRECISION} \
            -m lerobot.scripts.lerobot_train \
            --job_name="dit_lora_${LORA_CONFIG}_${DATASET}_s${SEED}" \
            --output_dir="./outputs/train/dit_lora_${LORA_CONFIG}_${DATASET}_s${SEED}" \
            --dataset.repo_id="continuallearning/${DATASET}" \
            --policy.type=dit \
            --policy.push_to_hub=true \
            --policy.repo_id="continuallearning/dit_lora_${LORA_CONFIG}_${DATASET}_seed${SEED}" \
            --batch_size=128 \
            --num_workers=16 \
            --steps=${STEPS} \
            --seed=${SEED} \
            --eval_freq=0 \
            --save_freq=${SAVE_FREQ} \
            --log_freq=100 \
            --wandb.enable=true \
            --wandb.disable_artifact=true \
            --wandb.project=clare_rebuttal \
            --wandb.entity=470620104-technical-university-of-munich
    done
done

echo "All DiT LoRA finetuning runs completed!"
