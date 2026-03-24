#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on dataset episodes by comparing predicted vs ground truth actions.

Instead of running a gym environment, this script iterates through dataset episodes
frame-by-frame, feeds observations to the policy via select_action(), and compares
the predicted actions against ground truth. Results are plotted and optionally logged
to WandB.

Usage:
```
lerobot-eval-on-dataset \
    --policy.path=outputs/train/task0/checkpoints/last/pretrained_model \
    --dataset.repo_id=lerobot/libero_10_subtask \
    --dataset.episodes="[0,1,2,3,4]" \
    --output_dir=outputs/eval_dataset/task0 \
    --wandb.enable=true --wandb.project=clare-eval
```

With PEFT:
```
PEFT_CONFIG_PATH=configs/peft/clare_dit \
PEFT_WEIGHT_PATH=outputs/train/task0/checkpoints/last/adapter \
lerobot-eval-on-dataset \
    --policy.path=outputs/train/task0/checkpoints/last/pretrained_model \
    --dataset.repo_id=lerobot/libero_10_subtask \
    --dataset.episodes="[0,1,2,3,4]" \
    --output_dir=outputs/eval_dataset/task0
```
"""

import datetime as dt
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

# Normalization override env vars (same semantics as lerobot_train.py).
# Priority: NORM_STATS_FILE > NORM_CHECKPOINT_PATH > pretrained/dataset stats.
REUSE_PRETRAINED_NORMALIZATION: bool = os.environ.get(
    "REUSE_PRETRAINED_NORMALIZATION", "true"
).lower() != "false"
NORM_CHECKPOINT_PATH: str | None = os.environ.get("NORM_CHECKPOINT_PATH")
NORM_STATS_FILE: str | None = os.environ.get("NORM_STATS_FILE")


def _load_normalizer_stats_from_checkpoint(checkpoint_path: str) -> dict:
    """Load normalization stats from a saved preprocessor checkpoint."""
    from lerobot.processor.normalize_processor import NormalizerProcessorStep
    from lerobot.processor.pipeline import DataProcessorPipeline

    preprocessor = DataProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename="policy_preprocessor.json",
    )
    for step in preprocessor.steps:
        if isinstance(step, NormalizerProcessorStep):
            return step.stats
    raise ValueError(
        f"No normalizer_processor step found in checkpoint: {checkpoint_path}"
    )


def parse_episodes(spec: str, total: int) -> list[int]:
    """Parse an episode specification string into a list of episode indices.

    Supported formats:
        "all"       → all episodes [0, total)
        "random"    → all episodes in random order
        "random:5"  → 5 random episodes
        "0,1,2"     → explicit list
        "0-10"      → range [0, 10]
        "0-10,15,20-25" → mixed
    """
    spec = spec.strip()
    if spec == "all" or spec == "":
        return list(range(total))
    if spec.startswith("random"):
        indices = list(range(total))
        random.shuffle(indices)
        if ":" in spec:
            n = int(spec.split(":")[1])
            indices = indices[:n]
        return indices

    indices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return [i for i in indices if 0 <= i < total]


@dataclass
class EvalOnDatasetConfig:
    dataset: DatasetConfig
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    episodes: str = "all"
    wandb: WandBConfig = field(default_factory=WandBConfig)
    rename_map: dict[str, str] = field(default_factory=dict)
    tolerance_s: float = 1e-4

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        else:
            logging.warning("No pretrained path provided, policy will be built from scratch.")

        if not self.job_name:
            self.job_name = f"eval_dataset_{self.policy.type if self.policy else 'scratch'}"

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval_on_dataset") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def load_dataset(cfg: EvalOnDatasetConfig) -> LeRobotDataset:
    # Load WITHOUT delta_timestamps so each item is a single frame.
    # select_action() handles observation history queuing internally.
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=None,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def load_policy_and_processors(cfg: EvalOnDatasetConfig, dataset: LeRobotDataset):
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # PEFT support
    peft_config_path = os.environ.get("PEFT_CONFIG_PATH")
    peft_weight_path = os.environ.get("PEFT_WEIGHT_PATH")
    if peft_config_path:
        from peft import PeftConfig as PeftLibConfig
        from peft import PeftModel

        logging.info(f"Loading PEFT config from {peft_config_path}")
        peft_config = PeftLibConfig.from_pretrained(peft_config_path)
        policy = policy.wrap_with_peft(peft_config=peft_config)

        if peft_weight_path:
            logging.info(f"Loading PEFT weights from {peft_weight_path}")
            # The wrapped policy's inner model is a PeftModel
            peft_model = None
            for module in policy.modules():
                if isinstance(module, PeftModel):
                    peft_model = module
                    break
            if peft_model is not None:
                from safetensors.torch import load_file

                adapter_file = Path(peft_weight_path) / "adapter_model.safetensors"
                if adapter_file.exists():
                    state_dict = load_file(str(adapter_file))
                    # Load directly into the PeftModel. We bypass set_peft_model_state_dict
                    # because CLARE uses nn.ModuleDict({adapter_name: ModuleList}) which
                    # keeps the adapter name ("default") as part of the module hierarchy.
                    # The generic PEFT save path (remove_adapter_name) is a no-op for these
                    # keys, but the generic load path (_insert_adapter_name_into_state_dict)
                    # erroneously duplicates "default", causing all keys to be silently
                    # dropped by load_state_dict(strict=False).
                    load_result = peft_model.load_state_dict(state_dict, strict=False)
                    missing_clare = [k for k in load_result.missing_keys if "clare_" in k]
                    if missing_clare:
                        logging.warning(f"Missing CLARE keys after load: {missing_clare[:5]}...")
                    if load_result.unexpected_keys:
                        logging.warning(f"Unexpected keys after load: {load_result.unexpected_keys[:5]}...")
                    logging.info("PEFT weights loaded successfully.")
                else:
                    logging.warning(f"No adapter_model.safetensors found at {peft_weight_path}")

    policy.eval()

    # Determine normalization stats.
    # Priority: NORM_STATS_FILE > NORM_CHECKPOINT_PATH > pretrained/dataset stats.
    _norm_stats = None
    if NORM_STATS_FILE:
        logging.info(f"Loading normalizer stats from NORM_STATS_FILE: {NORM_STATS_FILE}")
        with open(NORM_STATS_FILE) as f:
            _norm_stats = json.load(f)
    elif NORM_CHECKPOINT_PATH:
        logging.info(f"Loading normalizer stats from NORM_CHECKPOINT_PATH: {NORM_CHECKPOINT_PATH}")
        _norm_stats = _load_normalizer_stats_from_checkpoint(NORM_CHECKPOINT_PATH)
    elif not (cfg.policy.pretrained_path and REUSE_PRETRAINED_NORMALIZATION):
        _norm_stats = dataset.meta.stats

    # When _norm_stats is None the pretrained checkpoint's own stats are preserved
    # (no override passed to make_pre_post_processors).
    norm_stats = _norm_stats if _norm_stats is not None else dataset.meta.stats

    device = str(policy.config.device)
    preprocessor_overrides = {
        "device_processor": {"device": device},
        "normalizer_processor": {
            "stats": norm_stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "stats": norm_stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        },
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=norm_stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    return policy, preprocessor, postprocessor


def evaluate_episode(
    dataset: LeRobotDataset,
    ep_idx: int,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
    """Evaluate a single episode frame-by-frame, returning GT and predicted actions."""
    ep = dataset.meta.episodes[ep_idx]
    ep_start = ep["dataset_from_index"]
    ep_end = ep["dataset_to_index"]
    num_frames = ep_end - ep_start

    policy.reset()

    gt_actions = []
    pred_actions = []

    for frame_idx in tqdm(range(ep_start, ep_end), desc=f"Episode {ep_idx}", leave=False):
        item = dataset[frame_idx]

        # Extract ground truth action (single frame, shape [action_dim])
        gt_action = item[ACTION].clone()
        gt_actions.append(gt_action.cpu())

        # Build observation batch: keep only observation keys + task, add batch dim
        obs_batch = {}
        for key, val in item.items():
            if key.startswith("observation.") or key == "task" or key == "task_index":
                if isinstance(val, torch.Tensor):
                    # Ensure at least 1D before adding batch dim (scalars like gripper)
                    if val.dim() == 0:
                        val = val.unsqueeze(0)
                    obs_batch[key] = val.unsqueeze(0)
                else:
                    obs_batch[key] = val

        obs_batch = preprocessor(obs_batch)

        # Remove 'action' (set to None by preprocessor) so populate_queues inside
        # select_action doesn't fill the action queue with None values.
        obs_only = {
            k: v for k, v in obs_batch.items()
            if k.startswith("observation.") or k == "task" or k == "task_index"
        }

        # Force fresh forward pass every frame (bypass action chunk cache)
        if hasattr(policy, "_queues") and policy._queues is not None:
            policy._queues["action"].clear()

        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp else nullcontext()
        with torch.inference_mode(), amp_ctx:
            pred_action = policy.select_action(obs_only)

        # Postprocess to get unnormalized actions
        pred_action = postprocessor(pred_action)
        pred_actions.append(pred_action.squeeze(0).cpu())

    gt_actions = torch.stack(gt_actions)  # [T, action_dim]
    pred_actions = torch.stack(pred_actions)  # [T, action_dim]

    # Metrics
    mse_per_dim = ((pred_actions - gt_actions) ** 2).mean(dim=0)
    l1_per_dim = (pred_actions - gt_actions).abs().mean(dim=0)

    return {
        "ep_idx": ep_idx,
        "num_frames": num_frames,
        "gt_actions": gt_actions,
        "pred_actions": pred_actions,
        "mse_per_dim": mse_per_dim,
        "l1_per_dim": l1_per_dim,
        "mse": mse_per_dim.mean().item(),
        "l1": l1_per_dim.mean().item(),
    }


ACTION_DIM_NAMES = ["x", "y", "z", "roll", "yaw", "pitch", "gripper"]


def _dim_name(dim_i: int) -> str:
    if dim_i < len(ACTION_DIM_NAMES):
        return ACTION_DIM_NAMES[dim_i]
    return f"dim_{dim_i}"


def plot_episode(result: dict, output_dir: Path) -> plt.Figure:
    """Create a plot comparing GT vs predicted actions for one episode."""
    gt = result["gt_actions"].numpy()
    pred = result["pred_actions"].numpy()
    ep_idx = result["ep_idx"]
    action_dim = gt.shape[1]

    ncols = min(4, action_dim)
    nrows = math.ceil(action_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    fig.suptitle(f"Episode {ep_idx} — MSE: {result['mse']:.4f}, L1: {result['l1']:.4f}")

    timesteps = range(gt.shape[0])
    for dim_i in range(action_dim):
        row, col = divmod(dim_i, ncols)
        ax = axes[row][col]
        ax.plot(timesteps, gt[:, dim_i], "b-", linewidth=1, label="GT")
        ax.plot(timesteps, pred[:, dim_i], "r--", linewidth=1, label="Pred")
        ax.set_title(f"{_dim_name(dim_i)} (MSE: {result['mse_per_dim'][dim_i]:.4f})")
        ax.set_xlabel("Step")
        if dim_i == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for dim_i in range(action_dim, nrows * ncols):
        row, col = divmod(dim_i, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()

    plot_path = output_dir / "plots" / f"episode_{ep_idx}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)

    return fig


def plot_episode_metrics_bar(all_results: list[dict], output_dir: Path) -> tuple[plt.Figure, plt.Figure]:
    """Create per-episode MSE and L1 bar charts (x-axis = episode index)."""
    ep_indices = [r["ep_idx"] for r in all_results]
    mse_vals = [r["mse"] for r in all_results]
    l1_vals = [r["l1"] for r in all_results]

    fig_mse, ax_mse = plt.subplots(figsize=(max(8, len(ep_indices) * 0.4), 4))
    ax_mse.bar(ep_indices, mse_vals, color="steelblue")
    ax_mse.set_xlabel("Episode")
    ax_mse.set_ylabel("MSE")
    ax_mse.set_title("MSE per Episode")
    ax_mse.set_xticks(ep_indices)
    fig_mse.tight_layout()

    fig_l1, ax_l1 = plt.subplots(figsize=(max(8, len(ep_indices) * 0.4), 4))
    ax_l1.bar(ep_indices, l1_vals, color="coral")
    ax_l1.set_xlabel("Episode")
    ax_l1.set_ylabel("L1")
    ax_l1.set_title("L1 per Episode")
    ax_l1.set_xticks(ep_indices)
    fig_l1.tight_layout()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_mse.savefig(plots_dir / "mse_per_episode.png", dpi=150)
    fig_l1.savefig(plots_dir / "l1_per_episode.png", dpi=150)

    return fig_mse, fig_l1


def plot_summary(all_results: list[dict], output_dir: Path) -> plt.Figure:
    """Create a summary bar chart of per-dimension MSE/L1 averaged across episodes."""
    mse_stack = torch.stack([r["mse_per_dim"] for r in all_results])  # [N, action_dim]
    l1_stack = torch.stack([r["l1_per_dim"] for r in all_results])
    avg_mse = mse_stack.mean(dim=0).numpy()
    avg_l1 = l1_stack.mean(dim=0).numpy()
    action_dim = len(avg_mse)
    dim_labels = [_dim_name(i) for i in range(action_dim)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(dim_labels, avg_mse, color="steelblue")
    ax1.set_xlabel("Action Dimension")
    ax1.set_ylabel("MSE")
    ax1.set_title("Average MSE per Dimension")

    ax2.bar(dim_labels, avg_l1, color="coral")
    ax2.set_xlabel("Action Dimension")
    ax2.set_ylabel("L1")
    ax2.set_title("Average L1 per Dimension")

    fig.tight_layout()

    plot_path = output_dir / "plots" / "summary.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)

    return fig


def plot_action_dim_figure(
    gt: np.ndarray,
    pred: np.ndarray,
    dim_name: str,
    ep_idx: int,
    mse: float,
    y_min: float | None = None,
    y_max: float | None = None,
) -> plt.Figure:
    """Create a single-axis figure for one action dimension (GT vs predicted).

    *y_min* / *y_max* come from normalization bounds and set the y-axis range.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    timesteps = np.arange(len(gt))
    ax.plot(timesteps, gt, "b-", linewidth=1, label="GT")
    ax.plot(timesteps, pred, "r--", linewidth=1, label="Pred")
    ax.set_title(f"Episode {ep_idx} — {dim_name} (MSE: {mse:.4f})")
    ax.set_xlabel("Step")
    ax.set_ylabel(dim_name)
    ax.legend(fontsize=7)

    if y_min is not None and y_max is not None:
        margin = (y_max - y_min) * 0.05
        ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    return fig


def eval_on_dataset(cfg: EvalOnDatasetConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.seed is not None:
        set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    # WandB init
    wandb_run = None
    if cfg.wandb.enable:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.job_name,
            notes=cfg.wandb.notes,
            dir=str(output_dir),
            config=asdict(cfg),
            mode=cfg.wandb.mode if cfg.wandb.mode in ["online", "offline", "disabled"] else "online",
        )
        logging.info(f"WandB run: {wandb_run.get_url()}")

    logging.info("Loading dataset.")
    dataset = load_dataset(cfg)
    logging.info(f"Dataset: {dataset}")

    logging.info("Loading policy and processors.")
    policy, preprocessor, postprocessor = load_policy_and_processors(cfg, dataset)

    # Determine which episodes to evaluate
    total_episodes = len(dataset.meta.episodes)
    episode_indices = parse_episodes(cfg.episodes, total_episodes)
    logging.info(f"Evaluating {len(episode_indices)}/{total_episodes} episodes: {episode_indices}")

    # Extract normalization bounds for action dimensions (used for local plots)
    action_stats = dataset.meta.stats.get("action", {})
    action_norm_mode = policy.config.normalization_mapping.get("ACTION", "IDENTITY")
    action_bounds = None  # (min_per_dim, max_per_dim) arrays or None
    if action_norm_mode == "MIN_MAX" and "min" in action_stats and "max" in action_stats:
        action_bounds = (action_stats["min"], action_stats["max"])
    elif action_norm_mode == "QUANTILES" and "q01" in action_stats and "q99" in action_stats:
        action_bounds = (action_stats["q01"], action_stats["q99"])
    elif action_norm_mode == "QUANTILE10" and "q10" in action_stats and "q90" in action_stats:
        action_bounds = (action_stats["q10"], action_stats["q90"])
    elif action_norm_mode == "MEAN_STD" and "min" in action_stats and "max" in action_stats:
        action_bounds = (action_stats["min"], action_stats["max"])

    # Define wandb custom step metrics (one x-axis per episode)
    if wandb_run is not None:
        import wandb

        for ep_idx in episode_indices:
            step_key = f"episode_{ep_idx}/_step"
            wandb.define_metric(step_key, hidden=True)
            wandb.define_metric(f"episode_{ep_idx}/*", step_metric=step_key)

    all_results = []
    for ep_idx in tqdm(episode_indices, desc="Evaluating episodes"):
        result = evaluate_episode(
            dataset=dataset,
            ep_idx=ep_idx,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            use_amp=getattr(cfg.policy, "use_amp", False),
        )
        logging.info(f"Episode {ep_idx}: MSE={result['mse']:.4f}, L1={result['l1']:.4f}")
        all_results.append(result)

        gt = result["gt_actions"].numpy()
        pred = result["pred_actions"].numpy()
        action_dim = gt.shape[1]
        num_frames = result["num_frames"]

        # ---- wandb: raw scalars, frame-by-frame ----
        if wandb_run is not None:
            import wandb

            for t in range(num_frames):
                log_dict = {f"episode_{ep_idx}/_step": t}
                for dim_i in range(action_dim):
                    dname = _dim_name(dim_i)
                    log_dict[f"episode_{ep_idx}/{dname}_gt"] = float(gt[t, dim_i])
                    log_dict[f"episode_{ep_idx}/{dname}_pred"] = float(pred[t, dim_i])
                wandb.log(log_dict)

        # ---- Local: per-dimension matplotlib figures ----
        ep_plots_dir = output_dir / "plots" / f"episode_{ep_idx}"
        ep_plots_dir.mkdir(parents=True, exist_ok=True)

        for dim_i in range(action_dim):
            dname = _dim_name(dim_i)
            y_min = float(action_bounds[0][dim_i]) if action_bounds is not None else None
            y_max = float(action_bounds[1][dim_i]) if action_bounds is not None else None

            fig = plot_action_dim_figure(
                gt[:, dim_i], pred[:, dim_i], dname, ep_idx,
                result["mse_per_dim"][dim_i].item(), y_min, y_max,
            )
            fig.savefig(ep_plots_dir / f"{dname}.png", dpi=150)
            plt.close(fig)

    # ---- Summary ----
    summary_fig = plot_summary(all_results, output_dir)
    fig_mse, fig_l1 = plot_episode_metrics_bar(all_results, output_dir)
    plt.close(summary_fig)
    plt.close(fig_mse)
    plt.close(fig_l1)

    overall_mse = sum(r["mse"] for r in all_results) / len(all_results)
    overall_l1 = sum(r["l1"] for r in all_results) / len(all_results)

    avg_mse_per_dim = torch.stack([r["mse_per_dim"] for r in all_results]).mean(dim=0)
    avg_l1_per_dim = torch.stack([r["l1_per_dim"] for r in all_results]).mean(dim=0)

    metrics = {
        "overall_mse": overall_mse,
        "overall_l1": overall_l1,
        "mse_per_dim": avg_mse_per_dim.tolist(),
        "l1_per_dim": avg_l1_per_dim.tolist(),
        "per_episode": [
            {"episode": r["ep_idx"], "mse": r["mse"], "l1": r["l1"], "num_frames": r["num_frames"]}
            for r in all_results
        ],
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {metrics_path}")

    if wandb_run is not None:
        import wandb

        summary_log: dict = {
            "summary/mse": overall_mse,
            "summary/l1": overall_l1,
        }
        for i, (mse_val, l1_val) in enumerate(zip(avg_mse_per_dim.tolist(), avg_l1_per_dim.tolist())):
            summary_log[f"summary/mse_{_dim_name(i)}"] = mse_val
            summary_log[f"summary/l1_{_dim_name(i)}"] = l1_val
        for r in all_results:
            eidx = r["ep_idx"]
            summary_log[f"summary/per_ep/mse_ep_{eidx}"] = r["mse"]
            summary_log[f"summary/per_ep/l1_ep_{eidx}"] = r["l1"]
        wandb.log(summary_log)
        wandb.finish()

    logging.info(
        colored("Evaluation complete.", "green", attrs=["bold"])
        + f" Overall MSE: {overall_mse:.4f}, L1: {overall_l1:.4f}"
    )

    return metrics


@parser.wrap()
def eval_on_dataset_cli(cfg: EvalOnDatasetConfig):
    eval_on_dataset(cfg)


def main():
    init_logging()
    register_third_party_plugins()
    eval_on_dataset_cli()


if __name__ == "__main__":
    main()
