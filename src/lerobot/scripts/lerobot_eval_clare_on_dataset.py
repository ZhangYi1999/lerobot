#!/usr/bin/env python
"""Evaluate a CLARE policy on dataset episodes with routing diagnostics.

Extends lerobot_eval_on_dataset with CLARE-specific logging.

Logging strategy:
  - **wandb**: raw scalar values logged step-by-step via ``wandb.log()``.
    Each episode is a section, each action dim / routing layer is a panel.
  - **Local**: matplotlib figures saved as PNG to ``output_dir/plots/``.

Usage:
    PEFT_CONFIG_PATH=... PEFT_WEIGHT_PATH=... \
    python -m lerobot.scripts.lerobot_eval_clare_on_dataset \
        --policy.path=... --dataset.repo_id=... \
        --episodes=all --output_dir=... --wandb.enable=true
"""

import json
import logging
import math
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from termcolor import colored  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lerobot.configs import parser
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.scripts.lerobot_eval_on_dataset import (
    EvalOnDatasetConfig,
    _dim_name,
    load_dataset,
    load_policy_and_processors,
    parse_episodes,
)


# ---------------------------------------------------------------------------
# CLARE layer helpers
# ---------------------------------------------------------------------------

def _get_clare_layers(policy):
    """Extract CLARELayer list from a PEFT-wrapped policy."""
    from peft import PeftModel

    for module in policy.modules():
        if isinstance(module, PeftModel):
            return module.base_model.adapter_layers
    return []


def _layer_name(layer) -> str:
    return f"{layer.layer_name}.{layer.layer_id}.{layer.base_layer_name}"


def _wandb_safe(name: str) -> str:
    """Sanitize layer name for wandb keys (dots break panels)."""
    return name.replace(".", "_")


def _snapshot_clare_routing(clare_layers) -> dict[str, int]:
    """Read routing decision (top-1 discriminator index) from all CLARELayers.

    Returns dict: layer_name -> top_1_idx (int).
    """
    snap = {}
    for layer in clare_layers:
        info = layer.info_dicts
        if not info or "top_1_idx_list" not in info:
            continue
        name = _layer_name(layer)
        top_1 = info["top_1_idx_list"][0] if info["top_1_idx_list"] else -1
        snap[name] = top_1
    return snap


# ---------------------------------------------------------------------------
# Plotting helpers (local PNG only — never uploaded to wandb)
# ---------------------------------------------------------------------------

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


def plot_routing_layer_figure(
    top_1_idx_list: list[int],
    layer_name: str,
    ep_idx: int,
    n_disc: int,
) -> plt.Figure:
    """Step-plot of routing decisions over timesteps for one CLARE layer."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    timesteps = np.arange(len(top_1_idx_list))
    ax.step(timesteps, top_1_idx_list, where="mid", linewidth=1, color="teal")
    ax.set_title(f"Episode {ep_idx} — Routing: {layer_name}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Disc. idx")
    ax.set_yticks(range(n_disc))
    ax.set_ylim(-0.5, n_disc - 0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------

def evaluate_episode_clare(
    dataset,
    ep_idx: int,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    clare_layers: list,
    use_amp: bool = False,
) -> dict:
    """Evaluate one episode, collecting per-frame CLARE routing decisions."""
    ep = dataset.meta.episodes[ep_idx]
    ep_start = ep["dataset_from_index"]
    ep_end = ep["dataset_to_index"]
    num_frames = ep_end - ep_start

    policy.reset()

    gt_actions = []
    pred_actions = []
    per_frame_top1: dict[str, list[int]] = defaultdict(list)

    for frame_idx in tqdm(range(ep_start, ep_end), desc=f"Episode {ep_idx}", leave=False):
        item = dataset[frame_idx]

        gt_action = item[ACTION].clone()
        gt_actions.append(gt_action.cpu())

        obs_batch = {}
        for key, val in item.items():
            if key.startswith("observation.") or key == "task" or key == "task_index":
                if isinstance(val, torch.Tensor):
                    if val.dim() == 0:
                        val = val.unsqueeze(0)
                    obs_batch[key] = val.unsqueeze(0)
                else:
                    obs_batch[key] = val

        obs_batch = preprocessor(obs_batch)
        obs_only = {
            k: v for k, v in obs_batch.items()
            if k.startswith("observation.") or k == "task" or k == "task_index"
        }

        # Force fresh forward pass — bypass action chunk cache so every frame
        # triggers a full model forward (needed for correct CLARE routing info).
        if hasattr(policy, "_queues") and policy._queues is not None:
            policy._queues["action"].clear()

        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp else nullcontext()
        with torch.inference_mode(), amp_ctx:
            pred_action = policy.select_action(obs_only)

        snap = _snapshot_clare_routing(clare_layers)
        for lname, top1 in snap.items():
            per_frame_top1[lname].append(top1)

        pred_action = postprocessor(pred_action)
        pred_actions.append(pred_action.squeeze(0).cpu())

    gt_actions = torch.stack(gt_actions)
    pred_actions = torch.stack(pred_actions)

    mse_per_dim = ((pred_actions - gt_actions) ** 2).mean(dim=0)
    l1_per_dim = (pred_actions - gt_actions).abs().mean(dim=0)

    clare_info = {}
    for ln in sorted(per_frame_top1.keys()):
        clare_info[ln] = {"top_1_idx": per_frame_top1[ln]}

    return {
        "ep_idx": ep_idx,
        "num_frames": num_frames,
        "gt_actions": gt_actions,
        "pred_actions": pred_actions,
        "mse_per_dim": mse_per_dim,
        "l1_per_dim": l1_per_dim,
        "mse": mse_per_dim.mean().item(),
        "l1": l1_per_dim.mean().item(),
        "clare_info": clare_info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def eval_clare_on_dataset(cfg: EvalOnDatasetConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.seed is not None:
        set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    # WandB
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

    logging.info("Loading policy and processors.")
    policy, preprocessor, postprocessor = load_policy_and_processors(cfg, dataset)

    clare_layers = _get_clare_layers(policy)
    layer_names = [_layer_name(l) for l in clare_layers]
    logging.info(f"Found {len(clare_layers)} CLARE layers: {layer_names}")

    # Determine n_disc from the first CLARE layer
    n_disc = 0
    if clare_layers:
        layer0 = clare_layers[0]
        n_disc = len(layer0.clare_discriminators[layer0.adapter_name])

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
    all_routing: dict[str, list[list[int]]] = defaultdict(list)

    for ep_idx in tqdm(episode_indices, desc="Evaluating episodes"):
        result = evaluate_episode_clare(
            dataset=dataset,
            ep_idx=ep_idx,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            clare_layers=clare_layers,
            use_amp=getattr(cfg.policy, "use_amp", False),
        )
        logging.info(f"Episode {ep_idx}: MSE={result['mse']:.4f}, L1={result['l1']:.4f}")
        all_results.append(result)

        clare_info = result.get("clare_info", {})
        for lname, ldata in clare_info.items():
            all_routing[lname].append(ldata["top_1_idx"])

        gt = result["gt_actions"].numpy()
        pred = result["pred_actions"].numpy()
        action_dim = gt.shape[1]
        num_frames = result["num_frames"]

        # ---- Path A: wandb (raw scalars, frame-by-frame) ----
        if wandb_run is not None:
            import wandb

            for t in range(num_frames):
                log_dict = {f"episode_{ep_idx}/_step": t}
                for dim_i in range(action_dim):
                    dname = _dim_name(dim_i)
                    log_dict[f"episode_{ep_idx}/{dname}_gt"] = float(gt[t, dim_i])
                    log_dict[f"episode_{ep_idx}/{dname}_pred"] = float(pred[t, dim_i])
                for lname, ldata in clare_info.items():
                    safe = _wandb_safe(lname)
                    log_dict[f"episode_{ep_idx}/routing_{safe}"] = ldata["top_1_idx"][t]
                wandb.log(log_dict)

        # ---- Path B: local matplotlib figures ----
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

        for lname, ldata in clare_info.items():
            safe = _wandb_safe(lname)
            fig = plot_routing_layer_figure(
                ldata["top_1_idx"], lname, ep_idx, max(n_disc, 1),
            )
            fig.savefig(ep_plots_dir / f"routing_{safe}.png", dpi=150)
            plt.close(fig)

    # ---- Summary ----
    overall_mse = sum(r["mse"] for r in all_results) / len(all_results)
    overall_l1 = sum(r["l1"] for r in all_results) / len(all_results)
    avg_mse_per_dim = torch.stack([r["mse_per_dim"] for r in all_results]).mean(dim=0)
    avg_l1_per_dim = torch.stack([r["l1_per_dim"] for r in all_results]).mean(dim=0)

    # Build per-layer routing counts
    routing_counts: dict[str, np.ndarray] = {}
    for lname in layer_names:
        if lname not in all_routing:
            continue
        counts = np.zeros(max(n_disc, 1), dtype=int)
        for ep_top1 in all_routing[lname]:
            for idx in ep_top1:
                if 0 <= idx < len(counts):
                    counts[idx] += 1
        routing_counts[lname] = counts

    if wandb_run is not None:
        import wandb

        summary_log: dict = {
            "summary/mse": overall_mse,
            "summary/l1": overall_l1,
        }
        for i, (mse_v, l1_v) in enumerate(
            zip(avg_mse_per_dim.tolist(), avg_l1_per_dim.tolist())
        ):
            summary_log[f"summary/mse_{_dim_name(i)}"] = mse_v
            summary_log[f"summary/l1_{_dim_name(i)}"] = l1_v

        for r in all_results:
            eidx = r["ep_idx"]
            summary_log[f"summary/per_ep/mse_ep_{eidx}"] = r["mse"]
            summary_log[f"summary/per_ep/l1_ep_{eidx}"] = r["l1"]

        for lname in layer_names:
            if lname not in routing_counts:
                continue
            safe = _wandb_safe(lname)
            counts = routing_counts[lname]
            total = counts.sum()

            for d in range(len(counts)):
                summary_log[f"summary/clare/{safe}/count_disc_{d}"] = int(counts[d])

            if total > 0:
                probs = counts / total
                entropy = -sum(p * math.log(p + 1e-12) for p in probs)
                summary_log[f"summary/clare/{safe}/dominant_disc"] = int(np.argmax(counts))
                summary_log[f"summary/clare/{safe}/routing_entropy"] = entropy

        wandb.log(summary_log)
        wandb.finish()

    # Save metrics
    clare_summary = {}
    for lname, counts in routing_counts.items():
        total = counts.sum()
        freq = (counts / total).tolist() if total > 0 else counts.tolist()
        freq_nz = [p for p in freq if p > 0]
        entropy = -sum(p * math.log(p + 1e-12) for p in freq_nz) if freq_nz else 0.0
        clare_summary[lname] = {
            "routing_freq": freq,
            "routing_entropy": entropy,
            "dominant_disc": int(np.argmax(counts)),
        }

    metrics = {
        "overall_mse": overall_mse,
        "overall_l1": overall_l1,
        "mse_per_dim": avg_mse_per_dim.tolist(),
        "l1_per_dim": avg_l1_per_dim.tolist(),
        "per_episode": [
            {"episode": r["ep_idx"], "mse": r["mse"], "l1": r["l1"], "num_frames": r["num_frames"]}
            for r in all_results
        ],
        "clare": clare_summary,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {metrics_path}")

    logging.info(
        colored("Evaluation complete.", "green", attrs=["bold"])
        + f" Overall MSE: {overall_mse:.4f}, L1: {overall_l1:.4f}"
    )
    if clare_summary:
        logging.info(colored("CLARE routing summary:", "cyan"))
        for lname, stats in clare_summary.items():
            logging.info(
                f"  {lname}: entropy={stats['routing_entropy']:.2f}, "
                f"dominant={stats['dominant_disc']}, "
                f"freq={[f'{f:.1%}' for f in stats['routing_freq']]}"
            )

    return metrics


@parser.wrap()
def eval_clare_on_dataset_cli(cfg: EvalOnDatasetConfig):
    eval_clare_on_dataset(cfg)


def main():
    init_logging()
    register_third_party_plugins()
    eval_clare_on_dataset_cli()


if __name__ == "__main__":
    main()
