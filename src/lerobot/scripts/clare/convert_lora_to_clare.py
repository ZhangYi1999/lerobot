#!/usr/bin/env python
"""Convert N standard PEFT LoRA checkpoints into a single CLARE checkpoint.

Each LoRA checkpoint (one per task, trained independently on the same frozen backbone)
is loaded and its weights are mapped into CLARE's LoRAFuncAdapter structure.
Discriminators are created (randomly initialized) for each task — train them separately
via `--phase=discriminator_only` in clare.py.

Usage:
    python -m lerobot.scripts.clare.convert_lora_to_clare \
        --lora_checkpoint_dirs dir0 dir1 dir2 \
        --clare_config_path configs/peft/clare_dit \
        --output_dir outputs/clare_converted \
        --policy.type=dit \
        --policy.pretrained_path=outputs/train/.../pretrained_model \
        --dataset.repo_id=continuallearning/real_0_put_bowl_filtered
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file as load_safetensors

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.utils.utils import init_logging

from peft import get_peft_model, PeftConfig
from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING


class PeftWrapperPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy


@dataclass
class ConvertConfig(TrainPipelineConfig):
    """Config for LoRA → CLARE conversion."""
    lora_checkpoint_dirs: list[str] = field(default_factory=list)
    clare_config_path: str = ""
    use_policy_training_preset: bool = False


def find_lora_file(checkpoint_dir: Path) -> Path:
    """Find the LoRA weights file in a checkpoint directory."""
    safetensors_path = checkpoint_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        return safetensors_path
    bin_path = checkpoint_dir / "adapter_model.bin"
    if bin_path.exists():
        return bin_path
    raise FileNotFoundError(
        f"No adapter_model.safetensors or adapter_model.bin found in {checkpoint_dir}"
    )


def load_lora_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load LoRA weights from a checkpoint directory."""
    weights_file = find_lora_file(checkpoint_dir)
    if weights_file.suffix == ".safetensors":
        return load_safetensors(str(weights_file))
    else:
        return torch.load(str(weights_file), map_location="cpu", weights_only=True)


def build_lora_to_clare_key_mapping(
    clare_layers: list,
    lora_state_dict: dict[str, torch.Tensor],
) -> dict[str, str]:
    """Build mapping from LoRA state dict keys to CLARE state dict keys.

    For each CLARELayer, inspects lora_module_name_list to determine the sub-module
    names CLARE expects, then matches against LoRA keys.

    Returns dict: {lora_key -> clare_key_suffix} where clare_key_suffix is relative
    to the CLARELayer's position in the model.
    """
    # Build the mapping from CLARE's perspective
    # Each CLARELayer wraps a target module (e.g., attn_modulate.scale)
    # The lora_module_name_list contains sub-module names within the wrapped module
    # For single nn.Linear targets, this is ["self"]

    mapping = {}

    for clare_layer in clare_layers:
        # Get the full path to this CLARELayer's target in the original model
        # We need to reconstruct the path that both LoRA and CLARE would use
        # CLARE wraps: base_model.model.{path_to_module}
        # LoRA wraps: base_model.model.{path_to_module} (same module, different wrapping)

        for sub_name in clare_layer.lora_module_name_list:
            sub_key = sub_name.replace(".", "_")

            for lora_ab, clare_ab in [("lora_A", "lora_a"), ("lora_B", "lora_b")]:
                # The LoRA key suffix for this sub-module
                if sub_name == "self":
                    # The CLARELayer wraps the nn.Linear directly
                    # LoRA key: {module_path}.lora_A.default.weight
                    lora_suffix = f"{lora_ab}.default.weight"
                else:
                    # The CLARELayer wraps a parent module containing sub-Linears
                    # LoRA key: {module_path}.{sub_name}.lora_A.default.weight
                    lora_suffix = f"{sub_name}.{lora_ab}.default.weight"

                clare_suffix = f"layer_wise_lora_adapters.{sub_key}.{clare_ab}.weight"

                # Store with the base_layer_name for matching
                mapping_key = (clare_layer.base_layer_name, clare_layer.layer_name, clare_layer.layer_id, lora_suffix)
                mapping[mapping_key] = clare_suffix

    return mapping


def map_lora_weights_to_clare(
    lora_sd: dict[str, torch.Tensor],
    clare_model_sd: dict[str, torch.Tensor],
    clare_layers: list,
    adapter_idx: int,
) -> dict[str, torch.Tensor]:
    """Map LoRA state dict keys to CLARE state dict keys and copy weights.

    Args:
        lora_sd: Standard PEFT LoRA state dict (keys without base_model.model. prefix)
        clare_model_sd: Current CLARE model state dict (full keys)
        clare_layers: List of CLARELayer instances
        adapter_idx: Index of the adapter slot to fill

    Returns:
        Updated CLARE state dict with LoRA weights copied in
    """
    mapped_count = 0
    unmapped_lora_keys = set(lora_sd.keys())

    for clare_layer in clare_layers:
        for sub_name in clare_layer.lora_module_name_list:
            sub_key = sub_name.replace(".", "_")

            for lora_ab, clare_ab in [("lora_A", "lora_a"), ("lora_B", "lora_b")]:
                # Find the matching LoRA key
                if sub_name == "self":
                    lora_suffix = f".{lora_ab}.default.weight"
                else:
                    lora_suffix = f".{sub_name}.{lora_ab}.default.weight"

                # Search for matching LoRA key
                lora_key = None
                for k in lora_sd:
                    if k.endswith(lora_suffix):
                        # Verify this key corresponds to the right CLARE layer
                        # by checking the module path matches
                        lora_key = k
                        break

                if lora_key is None:
                    logging.warning(
                        f"No LoRA key found for CLARELayer "
                        f"{clare_layer.layer_name}.{clare_layer.layer_id} "
                        f"sub_module={sub_name} {lora_ab}"
                    )
                    continue

                # Find the matching CLARE key in the model state dict
                clare_adapter_key = None
                clare_pattern = (
                    f".clare_func_adapters.default.{adapter_idx}"
                    f".layer_wise_lora_adapters.{sub_key}.{clare_ab}.weight"
                )

                for k in clare_model_sd:
                    if k.endswith(clare_pattern):
                        clare_adapter_key = k
                        break

                if clare_adapter_key is None:
                    logging.warning(
                        f"No CLARE key found matching pattern *{clare_pattern}"
                    )
                    continue

                # Verify shapes match
                lora_weight = lora_sd[lora_key]
                clare_weight = clare_model_sd[clare_adapter_key]
                if lora_weight.shape != clare_weight.shape:
                    raise ValueError(
                        f"Shape mismatch: LoRA {lora_key} {lora_weight.shape} vs "
                        f"CLARE {clare_adapter_key} {clare_weight.shape}"
                    )

                # Copy weight
                clare_model_sd[clare_adapter_key] = lora_weight.clone()
                unmapped_lora_keys.discard(lora_key)
                mapped_count += 1
                logging.debug(f"  {lora_key} -> {clare_adapter_key}")

    if unmapped_lora_keys:
        logging.warning(
            f"Unmapped LoRA keys (not in CLARE targets): {unmapped_lora_keys}"
        )

    logging.info(f"Mapped {mapped_count} weight tensors for adapter {adapter_idx}")
    return clare_model_sd


@parser.wrap()
def convert(cfg: ConvertConfig):
    init_logging()

    if not cfg.lora_checkpoint_dirs:
        raise ValueError("--lora_checkpoint_dirs must specify at least one directory")
    if not cfg.clare_config_path:
        raise ValueError("--clare_config_path must be specified")

    lora_dirs = [Path(d) for d in cfg.lora_checkpoint_dirs]
    n_tasks = len(lora_dirs)
    logging.info(f"Converting {n_tasks} LoRA checkpoints to CLARE format")

    # Validate all checkpoint dirs exist
    for d in lora_dirs:
        find_lora_file(d)  # raises if not found

    # ---- Step 1: Create base policy ----
    logging.info("Loading base policy")
    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()

    # ---- Step 2: Create CLARE model shell ----
    logging.info(f"Creating CLARE model from config: {cfg.clare_config_path}")
    peft_wrapper_policy = PeftWrapperPolicy(policy=policy)

    clare_peft_cfg = PeftConfig.from_pretrained(cfg.clare_config_path)
    clare_peft_cfg.inference_mode = False
    peft_policy = get_peft_model(peft_wrapper_policy, clare_peft_cfg)
    peft_config = peft_policy.peft_config["default"]

    clare_layers = peft_policy.base_model.adapter_layers

    logging.info(f"Created {len(clare_layers)} CLARELayers")
    for cl in clare_layers:
        logging.info(
            f"  {cl.layer_name}.{cl.layer_id}.{cl.base_layer_name}: "
            f"lora_modules={cl.lora_module_name_list if cl.use_lora else 'N/A'}"
        )

    # ---- Step 3: Add adapters and discriminators for each task ----
    for task_id in range(n_tasks):
        logging.info(f"Adding adapter + discriminator for task {task_id}")
        for clare_layer in clare_layers:
            clare_layer.add_adapter_and_discriminator(task_id)
            key = f"{clare_layer.layer_name}.{clare_layer.layer_id}"
            peft_config.structure[key] = [
                clare_layer.num_adapters,
                clare_layer.num_discriminators,
            ]

    peft_config.num_learned_task = n_tasks

    # ---- Step 4: Load LoRA weights into CLARE adapters ----
    # Get the full CLARE model state dict
    clare_model_sd = peft_policy.state_dict()

    for task_id, lora_dir in enumerate(lora_dirs):
        logging.info(f"Loading LoRA checkpoint for task {task_id}: {lora_dir}")
        lora_sd = load_lora_state_dict(lora_dir)

        logging.info(f"  LoRA state dict has {len(lora_sd)} keys")
        if lora_sd:
            sample_key = next(iter(lora_sd))
            logging.info(f"  Sample key: {sample_key}")

        clare_model_sd = map_lora_weights_to_clare(
            lora_sd, clare_model_sd, clare_layers, adapter_idx=task_id
        )

    # ---- Step 5: Load mapped weights back into model ----
    peft_policy.load_state_dict(clare_model_sd)

    # ---- Step 6: Save CLARE checkpoint ----
    output_dir = Path(cfg.output_dir)
    adapter_output = output_dir / "adapter"
    adapter_output.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving CLARE checkpoint to {adapter_output}")
    peft_policy.save_pretrained(str(adapter_output))

    # Also save the base policy for reference
    pretrained_output = output_dir / "pretrained_model"
    pretrained_output.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(pretrained_output))

    logging.info(f"Conversion complete!")
    logging.info(f"  CLARE adapter: {adapter_output}")
    logging.info(f"  Base policy:   {pretrained_output}")
    logging.info(f"  Tasks:         {n_tasks}")
    logging.info(f"  CLARELayers:   {len(clare_layers)}")
    logging.info(f"  Structure:     {peft_config.structure}")

    # Print summary of what to do next
    logging.info("")
    logging.info("Next step: Train discriminators per task:")
    for task_id in range(n_tasks):
        logging.info(
            f"  python -m lerobot.scripts.clare.clare "
            f"--phase=discriminator_only "
            f"--peft_weight_path={adapter_output} "
            f"--discriminator_task_id={task_id} "
            f"--dataset.repo_id=... --dataset.episodes=..."
        )


if __name__ == "__main__":
    convert()
