#!/usr/bin/env python
"""Visualize and compare normalization stats across multiple datasets.

Reads stats.json files and plots the implied distributions:
  - gaussian mode: N(mean, std) PDF curves
  - minmax mode: U(min, max) uniform rectangles
  - quantile mode: U(q01, q99) with percentile markers
  - auto mode: gaussian if std available, else minmax

Usage:
    python -m lerobot.scripts.clare.compare_dataset_stats \
        stats1.json stats2.json stats3.json \
        --labels "task0" "task1" "task2" \
        --mode gaussian \
        --output comparison.png
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_stats(path: str) -> dict[str, dict[str, list]]:
    """Load stats.json and return nested {feature: {stat_name: values}}."""
    with open(path) as f:
        raw = json.load(f)

    # Handle both flat ("action/mean": [...]) and nested ({"action": {"mean": [...]}}) formats
    if any("/" in k for k in raw):
        # Flat format — unflatten
        nested: dict[str, dict[str, list]] = {}
        for flat_key, values in raw.items():
            parts = flat_key.rsplit("/", 1)
            if len(parts) != 2:
                continue
            feature, stat_name = parts
            nested.setdefault(feature, {})[stat_name] = values
        return nested
    return raw


def is_image_key(stats: dict[str, list]) -> bool:
    """Heuristic: image stats have shape (3,1,1) or similar non-flat structures."""
    for stat_name in ("mean", "min"):
        if stat_name in stats:
            val = stats[stat_name]
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
                return True
    return False


def get_non_image_keys(all_stats: list[dict[str, dict[str, list]]]) -> list[str]:
    """Collect all feature keys across datasets, excluding image keys."""
    keys = set()
    for stats in all_stats:
        for feature, fstats in stats.items():
            if not is_image_key(fstats):
                keys.add(feature)
    return sorted(keys)


def get_ndims(feature_stats_list: list[dict[str, list] | None]) -> int:
    """Get number of dimensions for a feature across all datasets."""
    for fstats in feature_stats_list:
        if fstats is None:
            continue
        for stat_name in ("mean", "min", "q01"):
            if stat_name in fstats:
                val = fstats[stat_name]
                if isinstance(val, list):
                    return len(val)
                return 1
    return 0


def plot_gaussian(ax, mean: float, std: float, color, label: str, alpha: float = 0.6):
    """Plot a Gaussian PDF curve."""
    if std < 1e-10:
        # Degenerate: plot a vertical line at the mean
        ax.axvline(mean, color=color, label=label, alpha=alpha, linewidth=2)
        return mean - 0.5, mean + 0.5

    lo = mean - 4 * std
    hi = mean + 4 * std
    x = np.linspace(lo, hi, 200)
    y = (1 / (std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    ax.plot(x, y, color=color, label=label, alpha=alpha, linewidth=1.5)
    ax.fill_between(x, y, color=color, alpha=alpha * 0.3)
    return lo, hi


def plot_uniform(ax, lo: float, hi: float, color, label: str, alpha: float = 0.6):
    """Plot a uniform distribution rectangle."""
    if abs(hi - lo) < 1e-10:
        ax.axvline(lo, color=color, label=label, alpha=alpha, linewidth=2)
        return lo - 0.5, lo + 0.5

    height = 1.0 / (hi - lo)
    ax.fill_between([lo, hi], 0, height, color=color, alpha=alpha * 0.4, label=label)
    ax.plot([lo, lo, hi, hi], [0, height, height, 0], color=color, alpha=alpha, linewidth=1.5)
    return lo, hi


def plot_quantile(
    ax, q01: float, q99: float, color, label: str,
    q10: float | None = None, q50: float | None = None, q90: float | None = None,
    alpha: float = 0.6,
):
    """Plot uniform U(q01, q99) with optional percentile markers."""
    lo, hi = plot_uniform(ax, q01, q99, color, label, alpha)
    for q_val, q_name in [(q10, "q10"), (q50, "q50"), (q90, "q90")]:
        if q_val is not None:
            ax.axvline(q_val, color=color, linestyle="--", alpha=alpha * 0.7, linewidth=1)
    return lo, hi


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and compare normalization stats from multiple stats.json files."
    )
    parser.add_argument("stats_files", nargs="+", help="Paths to stats.json files")
    parser.add_argument("--labels", nargs="+", default=None, help="Display names for each dataset")
    parser.add_argument(
        "--mode", default="auto", choices=["gaussian", "minmax", "quantile", "auto"],
        help="Distribution type to plot (default: auto)",
    )
    parser.add_argument("--output", default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    n_datasets = len(args.stats_files)
    labels = args.labels or [Path(p).parent.name for p in args.stats_files]
    if len(labels) < n_datasets:
        labels.extend([f"dataset_{i}" for i in range(len(labels), n_datasets)])

    # Load all stats
    all_stats = [load_stats(p) for p in args.stats_files]

    # Get non-image feature keys
    feature_keys = get_non_image_keys(all_stats)
    if not feature_keys:
        print("No non-image feature keys found.")
        return

    # Collect (feature_key, ndims) pairs for subplot layout
    subplot_info: list[tuple[str, int]] = []
    for key in feature_keys:
        fstats_list = [s.get(key) for s in all_stats]
        ndims = get_ndims(fstats_list)
        if ndims > 0:
            subplot_info.append((key, ndims))

    total_subplots = sum(ndims for _, ndims in subplot_info)
    if total_subplots == 0:
        print("No plottable dimensions found.")
        return

    # Layout: use a reasonable number of columns
    ncols = min(6, total_subplots)
    nrows = math.ceil(total_subplots / ncols)

    colors = plt.cm.tab10.colors[:n_datasets]  # type: ignore[attr-defined]
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    subplot_idx = 0
    for key, ndims in subplot_info:
        fstats_list = [s.get(key) for s in all_stats]

        for dim in range(ndims):
            if subplot_idx >= len(axes_flat):
                break
            ax = axes_flat[subplot_idx]
            subplot_idx += 1

            x_lo_all, x_hi_all = float("inf"), float("-inf")

            for ds_idx, fstats in enumerate(fstats_list):
                if fstats is None:
                    continue

                color = colors[ds_idx % len(colors)]
                label = labels[ds_idx]
                mode = args.mode

                # Auto-detect mode
                if mode == "auto":
                    if "mean" in fstats and "std" in fstats:
                        mode = "gaussian"
                    elif "q01" in fstats and "q99" in fstats:
                        mode = "quantile"
                    elif "min" in fstats and "max" in fstats:
                        mode = "minmax"
                    else:
                        continue

                if mode == "gaussian" and "mean" in fstats and "std" in fstats:
                    mean_val = float(fstats["mean"][dim])
                    std_val = float(fstats["std"][dim])
                    lo, hi = plot_gaussian(ax, mean_val, std_val, color, label)
                    x_lo_all = min(x_lo_all, lo)
                    x_hi_all = max(x_hi_all, hi)

                elif mode == "minmax" and "min" in fstats and "max" in fstats:
                    min_val = float(fstats["min"][dim])
                    max_val = float(fstats["max"][dim])
                    lo, hi = plot_uniform(ax, min_val, max_val, color, label)
                    x_lo_all = min(x_lo_all, lo)
                    x_hi_all = max(x_hi_all, hi)

                elif mode == "quantile" and "q01" in fstats and "q99" in fstats:
                    q01 = float(fstats["q01"][dim])
                    q99 = float(fstats["q99"][dim])
                    q10 = float(fstats["q10"][dim]) if "q10" in fstats else None
                    q50 = float(fstats["q50"][dim]) if "q50" in fstats else None
                    q90 = float(fstats["q90"][dim]) if "q90" in fstats else None
                    lo, hi = plot_quantile(ax, q01, q99, color, label, q10, q50, q90)
                    x_lo_all = min(x_lo_all, lo)
                    x_hi_all = max(x_hi_all, hi)

            # Styling
            if ndims > 1:
                ax.set_title(f"{key}[{dim}]", fontsize=8)
            else:
                ax.set_title(key, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

            # Set x range with padding
            if x_lo_all < x_hi_all:
                margin = (x_hi_all - x_lo_all) * 0.1
                ax.set_xlim(x_lo_all - margin, x_hi_all + margin)

    # Hide unused subplots
    for i in range(subplot_idx, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Single legend for the whole figure
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper right", fontsize=8)

    fig.suptitle(f"Dataset Stats Comparison ({args.mode} mode)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    # Default output path: ./outputs/{label0}_{label1}...png
    output = args.output
    if output is None:
        output = f"./outputs/{'_'.join(labels)}.png"

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output}")

    # Save comparison data as JSON alongside the figure
    comparison_data: dict = {"labels": labels, "mode": args.mode, "features": {}}
    for key, ndims in subplot_info:
        feature_data: dict = {"ndims": ndims, "datasets": {}}
        for ds_idx, stats in enumerate(all_stats):
            fstats = stats.get(key)
            if fstats is None:
                continue
            ds_data: dict = {}
            for stat_name in ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99", "count"):
                if stat_name in fstats:
                    ds_data[stat_name] = fstats[stat_name]
            feature_data["datasets"][labels[ds_idx]] = ds_data
        comparison_data["features"][key] = feature_data

    json_output = Path(output).with_suffix(".json")
    with open(json_output, "w") as f:
        json.dump(comparison_data, f, indent=2, default=str)
    print(f"Saved data to {json_output}")


if __name__ == "__main__":
    main()
