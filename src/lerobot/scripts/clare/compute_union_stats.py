"""Compute union normalization stats across multiple datasets.

Usage:
    python -m lerobot.scripts.clare.compute_union_stats \
        --datasets continuallearning/real_0 continuallearning/real_1 ... \
        --output union_stats.json

    # Local paths also work:
    python -m lerobot.scripts.clare.compute_union_stats \
        --datasets /path/to/dataset_0 /path/to/dataset_1 ... \
        --output union_stats.json

The output JSON has the same format as LeRobot's meta/stats.json and can be
used with NORM_STATS_FILE env var in lerobot_train.py and CLARE scripts.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _to_list(v):
    """Convert numpy arrays / scalars to plain Python lists for JSON serialization."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [float(x) if isinstance(x, (int, float, np.floating, np.integer)) else x for x in v]
    if isinstance(v, (int, float, np.floating, np.integer)):
        return [float(v)]
    return v


def load_dataset_stats(dataset_id: str) -> dict:
    """Load stats from a Hub repo_id or local path.

    Returns the stats dict: {feature: {stat_name: np.ndarray}}.
    """
    root = Path(dataset_id)
    if root.is_dir():
        # Local path
        meta = LeRobotDatasetMetadata(repo_id=dataset_id, root=root)
    else:
        # Hub repo_id — downloads metadata only
        meta = LeRobotDatasetMetadata(repo_id=dataset_id)
    if meta.stats is None:
        raise ValueError(f"No stats found for dataset: {dataset_id}")
    return meta.stats


def compute_union_stats(all_stats: list[dict]) -> dict:
    """Merge stats from multiple datasets into a single union stats dict.

    For each feature present in any dataset:
      - mean:  weighted average by count
      - std:   combined std = sqrt(weighted_avg(var + mean^2) - combined_mean^2)
      - min:   element-wise min
      - max:   element-wise max
      - q01:   element-wise min (conservative lower bound)
      - q10:   element-wise min
      - q50:   weighted average (approximate)
      - q90:   element-wise max
      - q99:   element-wise max (conservative upper bound)
      - count: sum of counts
    """
    # Collect all feature keys
    all_features = set()
    for stats in all_stats:
        all_features.update(stats.keys())

    union = {}
    for feature in sorted(all_features):
        # Gather per-dataset stats for this feature
        entries = []
        for stats in all_stats:
            if feature in stats:
                entries.append(stats[feature])
        if not entries:
            continue

        # Get count per dataset (scalar or 1-element array)
        counts = []
        for e in entries:
            if "count" in e:
                c = e["count"]
                counts.append(float(c[0]) if hasattr(c, "__len__") else float(c))
            else:
                counts.append(1.0)
        total_count = sum(counts)
        weights = [c / total_count for c in counts]

        result = {}

        # --- mean: weighted average ---
        if all("mean" in e for e in entries):
            means = [np.asarray(e["mean"], dtype=np.float64) for e in entries]
            combined_mean = sum(w * m for w, m in zip(weights, means))
            result["mean"] = combined_mean

        # --- std: combined formula ---
        if all("std" in e for e in entries) and "mean" in result:
            stds = [np.asarray(e["std"], dtype=np.float64) for e in entries]
            means = [np.asarray(e["mean"], dtype=np.float64) for e in entries]
            # E[X^2] = var + mean^2, weighted average
            weighted_second_moment = sum(
                w * (s**2 + m**2) for w, s, m in zip(weights, stds, means)
            )
            combined_var = weighted_second_moment - result["mean"] ** 2
            # Clamp to avoid negative variance from floating point
            combined_var = np.maximum(combined_var, 0.0)
            result["std"] = np.sqrt(combined_var)

        # --- min/max: element-wise extremes ---
        if all("min" in e for e in entries):
            result["min"] = np.minimum.reduce(
                [np.asarray(e["min"], dtype=np.float64) for e in entries]
            )
        if all("max" in e for e in entries):
            result["max"] = np.maximum.reduce(
                [np.asarray(e["max"], dtype=np.float64) for e in entries]
            )

        # --- quantiles: conservative bounds ---
        if all("q01" in e for e in entries):
            result["q01"] = np.minimum.reduce(
                [np.asarray(e["q01"], dtype=np.float64) for e in entries]
            )
        if all("q10" in e for e in entries):
            result["q10"] = np.minimum.reduce(
                [np.asarray(e["q10"], dtype=np.float64) for e in entries]
            )
        if all("q50" in e for e in entries):
            # Weighted average as approximation
            q50s = [np.asarray(e["q50"], dtype=np.float64) for e in entries]
            result["q50"] = sum(w * q for w, q in zip(weights, q50s))
        if all("q90" in e for e in entries):
            result["q90"] = np.maximum.reduce(
                [np.asarray(e["q90"], dtype=np.float64) for e in entries]
            )
        if all("q99" in e for e in entries):
            result["q99"] = np.maximum.reduce(
                [np.asarray(e["q99"], dtype=np.float64) for e in entries]
            )

        # --- count: sum ---
        result["count"] = np.array([total_count])

        # Convert all to plain lists
        union[feature] = {k: _to_list(v) for k, v in result.items()}

    return union


def main():
    parser = argparse.ArgumentParser(
        description="Compute union normalization stats across multiple datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset repo_ids or local paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="union_stats.json",
        help="Output JSON file path (default: union_stats.json).",
    )
    args = parser.parse_args()

    logger.info(f"Loading stats from {len(args.datasets)} datasets...")
    all_stats = []
    for ds_id in args.datasets:
        logger.info(f"  Loading: {ds_id}")
        stats = load_dataset_stats(ds_id)
        all_stats.append(stats)

    logger.info("Computing union stats...")
    union = compute_union_stats(all_stats)

    # Pretty-print summary for action and observation.state
    for feature in ["action", "observation.state"]:
        if feature in union:
            s = union[feature]
            logger.info(f"\n{feature}:")
            if "mean" in s and "std" in s:
                for i, (m, sd) in enumerate(zip(s["mean"], s["std"])):
                    logger.info(f"  dim {i}: mean={m:.6f}, std={sd:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(union, f, indent=2)
    logger.info(f"\nSaved union stats to: {output_path}")


if __name__ == "__main__":
    main()
