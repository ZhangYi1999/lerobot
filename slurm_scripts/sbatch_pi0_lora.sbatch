#!/bin/bash
#SBATCH -J pi0_lora
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH -q mcml
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -o /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/slurm_logs/clare/pi0_lora.out
#SBATCH -e /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/slurm_logs/clare/pi0_lora.err

source ~/.bashrc

srun --ntasks-per-node=1 \
        --container-mounts="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/yi_zhang/lerobot:/workspace/lerobot,/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/yi_zhang/peft:/workspace/peft,/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/huggingface:/workspace/huggingface,/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/checkpoints/clare:/workspace/lerobot/outputs" \
        --container-image="${ENROOT}/clare.sqsh" \
        bash -c "cd /workspace/lerobot; \
        chmod +x ./bash_scripts/finetune_pi0_lora.sh; \
        ./bash_scripts/finetune_pi0_lora.sh"
