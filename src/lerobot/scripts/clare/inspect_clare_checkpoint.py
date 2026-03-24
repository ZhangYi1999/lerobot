"""Inspect metadata buffers (task_id, connected_adapter_indices, etc.) in a CLARE checkpoint."""

import argparse
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def fix_buffers(ckpt_path: str):
    """Set task_id, connected_adapter_indices, connected_adapter_task_id for all discriminators/adapters."""
    sf_path = str(Path(ckpt_path) / "adapter_model.safetensors")
    sd = load_file(sf_path)

    # Discover max index
    disc_pattern = re.compile(r"\.clare_discriminators\.default\.(\d+)\.")
    adapt_pattern = re.compile(r"\.clare_func_adapters\.default\.(\d+)\.")
    disc_ids = set()
    adapt_ids = set()
    for k in sd:
        m = disc_pattern.search(k)
        if m:
            disc_ids.add(int(m.group(1)))
        m = adapt_pattern.search(k)
        if m:
            adapt_ids.add(int(m.group(1)))

    num_tasks = max(max(disc_ids, default=-1), max(adapt_ids, default=-1)) + 1
    count = 0
    for task_id in range(num_tasks):
        for key in list(sd.keys()):
            disc_pre = f".clare_discriminators.default.{task_id}."
            adapt_pre = f".clare_func_adapters.default.{task_id}."
            if disc_pre in key:
                if key.endswith(".task_id"):
                    sd[key] = torch.tensor(task_id, dtype=torch.int64)
                    count += 1
                elif key.endswith(".connected_adapter_indices"):
                    sd[key] = torch.tensor(task_id, dtype=torch.int64)
                    count += 1
                elif key.endswith(".connected_adapter_task_id"):
                    sd[key] = torch.tensor(task_id, dtype=torch.int64)
                    count += 1
            elif adapt_pre in key and key.endswith(".task_id"):
                sd[key] = torch.tensor(task_id, dtype=torch.int64)
                count += 1

    save_file(sd, sf_path)
    print(f"Fixed {count} buffer values across {num_tasks} tasks in {ckpt_path}")


def inspect(ckpt_path: str):
    sd = load_file(str(Path(ckpt_path) / "adapter_model.safetensors"))

    # Collect metadata buffers
    meta_suffixes = (".task_id", ".connected_adapter_indices", ".connected_adapter_task_id",
                     ".num_batches_tracked", ".running_mean", ".running_std")

    disc_keys = sorted(k for k in sd if ".clare_discriminators." in k and k.endswith(meta_suffixes))
    adapt_keys = sorted(k for k in sd if ".clare_func_adapters." in k and k.endswith(meta_suffixes))

    # Extract unique layer prefixes (everything before .clare_*)
    layers = sorted(set(
        k.split(".clare_discriminators.")[0] for k in sd if ".clare_discriminators." in k
    ))

    print(f"Checkpoint: {ckpt_path}")
    print(f"Total keys: {len(sd)}")
    print(f"CLARE layers: {len(layers)}")
    print()

    # Show all CLARE layers in detail
    for layer in layers:
        print(f"=== {layer} ===")
        print()
        print("Discriminators:")
        for k in disc_keys:
            if k.startswith(f"{layer}.clare_discriminators."):
                short = k.split(".clare_discriminators.")[1]
                v = sd[k]
                print(f"  {short}: {v.item() if v.numel() == 1 else v.tolist()}")

        print()
        print("Func Adapters:")
        for k in adapt_keys:
            if k.startswith(f"{layer}.clare_func_adapters."):
                short = k.split(".clare_func_adapters.")[1]
                v = sd[k]
                print(f"  {short}: {v.item() if v.numel() == 1 else v.tolist()}")
        print()

    # Summary: check consistency across all layers
    print()
    print("=== Consistency check across all layers ===")
    issues = []
    for layer in layers:
        for suffix in (".task_id", ".connected_adapter_indices", ".connected_adapter_task_id"):
            vals = {}
            for k in sd:
                if k.startswith(f"{layer}.clare_discriminators.") and k.endswith(suffix):
                    idx = k.split(".clare_discriminators.default.")[1].split(".")[0]
                    vals[int(idx)] = sd[k].item()
            for idx, val in sorted(vals.items()):
                if val == -1:
                    issues.append(f"{layer} disc.{idx}{suffix} = -1 (unset)")

        adapt_vals = {}
        for k in sd:
            if k.startswith(f"{layer}.clare_func_adapters.") and k.endswith(".task_id"):
                idx = k.split(".clare_func_adapters.default.")[1].split(".")[0]
                adapt_vals[int(idx)] = sd[k].item()
        for idx, val in sorted(adapt_vals.items()):
            if val == -1:
                issues.append(f"{layer} adapter.{idx}.task_id = -1 (unset)")

    if issues:
        print(f"ISSUES FOUND ({len(issues)}):")
        for issue in issues[:20]:
            print(f"  {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("All metadata buffers are set correctly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to CLARE adapter checkpoint directory")
    parser.add_argument("--fix", action="store_true", help="Fix buffer values in-place")
    args = parser.parse_args()
    if args.fix:
        fix_buffers(args.checkpoint)
    inspect(args.checkpoint)
