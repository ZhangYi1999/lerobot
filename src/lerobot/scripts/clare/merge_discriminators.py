"""Merge separately-trained discriminator checkpoints into a single CLARE checkpoint.

Each discriminator training job only trains discriminator[task_id]. This script
takes N checkpoints (one per task) and merges them so that the output has all N
discriminators fully trained.
"""

import argparse
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")


def merge(task_checkpoint_dirs: list[str], output_dir: str):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Use task 0's checkpoint as base (has all adapters + disc 0 trained)
    base_path = Path(task_checkpoint_dirs[0])
    merged_sd = load_file(str(base_path / "adapter_model.safetensors"))
    logging.info(f"Base checkpoint: {base_path} ({len(merged_sd)} keys)")

    # For each task i, copy discriminator i's weights from task i's checkpoint
    for task_id, ckpt_dir in enumerate(task_checkpoint_dirs):
        ckpt_path = Path(ckpt_dir)
        sd = load_file(str(ckpt_path / "adapter_model.safetensors"))
        disc_prefix = f".clare_discriminators.default.{task_id}."
        count = 0
        for key in sd:
            if disc_prefix in key:
                merged_sd[key] = sd[key]
                count += 1
        logging.info(f"Task {task_id}: merged {count} discriminator keys from {ckpt_dir}")

    # Set metadata buffers for discriminators and adapters
    num_tasks = len(task_checkpoint_dirs)
    for task_id in range(num_tasks):
        for key in list(merged_sd.keys()):
            disc_pre = f".clare_discriminators.default.{task_id}."
            adapt_pre = f".clare_func_adapters.default.{task_id}."
            if disc_pre in key:
                if key.endswith(".task_id"):
                    merged_sd[key] = torch.tensor(task_id, dtype=torch.int64)
                elif key.endswith(".connected_adapter_indices"):
                    merged_sd[key] = torch.tensor(task_id, dtype=torch.int64)
                elif key.endswith(".connected_adapter_task_id"):
                    merged_sd[key] = torch.tensor(task_id, dtype=torch.int64)
            elif adapt_pre in key and key.endswith(".task_id"):
                merged_sd[key] = torch.tensor(task_id, dtype=torch.int64)
    logging.info(f"Set metadata buffers for {num_tasks} tasks")

    # Save merged checkpoint
    save_file(merged_sd, str(output / "adapter_model.safetensors"))
    # Copy adapter_config.json from base
    shutil.copy2(base_path / "adapter_config.json", output / "adapter_config.json")

    # Verify: check all discriminators have non-zero num_batches_tracked
    num_tasks = len(task_checkpoint_dirs)
    for task_id in range(num_tasks):
        batches_keys = [
            k for k in merged_sd
            if f".clare_discriminators.default.{task_id}.num_batches_tracked" in k
        ]
        vals = [merged_sd[k].item() for k in batches_keys]
        assert all(v > 0 for v in vals), (
            f"Disc {task_id} has untrained layers! "
            f"num_batches_tracked values: {vals}"
        )
        logging.info(
            f"Disc {task_id}: num_batches_tracked={vals[0]} "
            f"(verified across {len(batches_keys)} layers)"
        )

    logging.info(f"Merged checkpoint saved to {output} ({len(merged_sd)} keys)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task_checkpoints", nargs="+", required=True,
        help="Paths to per-task discriminator checkpoint dirs (order = task id)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for the merged checkpoint",
    )
    args = parser.parse_args()
    merge(args.task_checkpoints, args.output_dir)


if __name__ == "__main__":
    main()
