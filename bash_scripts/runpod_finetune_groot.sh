#!/bin/bash
set -e

# RunPod finetune_groot.sh — runs directly inside the pod (no docker run needed)
#
# Prerequisites:
#   1. Set WANDB_API_KEY and HF_TOKEN as RunPod pod environment variables
#   2. Clone repos onto network volume (first time only):
#        cd /runpod-volume
#        git clone https://github.com/ZhangYi1999/lerobot.git && cd lerobot && git checkout clare && cd ..
#        git clone <peft_lsy-repo> peft_lsy
#   3. Run:
#        bash /runpod-volume/lerobot/bash_scripts/runpod_finetune_groot.sh

VOLUME=/runpod-volume

# Source .env from network volume if it exists (fallback for secrets)
if [ -f "${VOLUME}/.env" ]; then
    source "${VOLUME}/.env"
fi

# Point caches to network volume
export HF_HOME=${VOLUME}/huggingface
export TORCH_HOME=${VOLUME}/torch

# Install code in editable mode (only needed once per pod start, fast if already installed)
pip install -e "${VOLUME}/lerobot[groot,libero]"
pip install -e "${VOLUME}/peft_lsy"

# Run training
cd ${VOLUME}/lerobot
bash bash_scripts/finetune_groot.sh
