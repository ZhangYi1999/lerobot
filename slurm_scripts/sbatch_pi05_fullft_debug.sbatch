#!/bin/bash
#SBATCH -J pi05_ft_dbg
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/slurm_logs/clare/pi05_fullft_debug.out
#SBATCH -e /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/slurm_logs/clare/pi05_fullft_debug.err

source ~/.bashrc

srun --ntasks-per-node=1 \
        --container-mounts="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/yi_zhang/lerobot:/workspace/lerobot,/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/yi_zhang/peft:/workspace/peft,/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/huggingface:/workspace/huggingface,/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/checkpoints/clare:/workspace/lerobot/outputs" \
        --container-image="${ENROOT}/clare.sqsh" \
        bash -c "cd /workspace/lerobot; \
        chmod +x ./bash_scripts/finetune_pi05_fullft.sh; \
        DEBUG=true ./bash_scripts/finetune_pi05_fullft.sh"
