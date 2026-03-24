#!/usr/bin/env python

"""Create an Experience Replay dataset by merging replay data with new data.

Supports recursive merging for continual learning:
  Step 1: dataset_0 (replay) + dataset_1 (new) → dataset_0_1
  Step 2: dataset_0_1 + dataset_2 → dataset_0_1_2
    - dataset_0 replay episodes preserved
    - dataset_1 episodes sampled as new replay
    - dataset_2 is the new data

Usage:
    python -m lerobot.scripts.clare.create_er_dataset \
        --existing_repo_id dataset_0_1 \
        --existing_root /path/to/dataset_0_1 \
        --new_repo_id dataset_2 \
        --new_root /path/to/dataset_2 \
        --output_repo_id dataset_0_1_2 \
        --output_root /path/to/output \
        --sample_n_episodes 50
"""

import argparse
import json
import logging
import random
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME

ER_META_FILENAME = "er_meta.json"


def load_er_meta(dataset_root: Path) -> dict | None:
    """Load er_meta.json from a dataset root, or return None if not found."""
    meta_path = dataset_root / ER_META_FILENAME
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def save_er_meta(output_root: Path, er_meta: dict):
    """Save er_meta.json to a dataset root."""
    meta_path = output_root / ER_META_FILENAME
    with open(meta_path, "w") as f:
        json.dump(er_meta, f, indent=2)
    logging.info(f"Saved ER metadata to {meta_path}")


def compute_er_meta(
    existing_meta: LeRobotDatasetMetadata,
    new_meta: LeRobotDatasetMetadata,
    existing_er_meta: dict | None,
    sample_n_episodes: int | None,
    seed: int = 42,
) -> dict:
    """Compute the er_meta.json for the merged dataset.

    After aggregate_datasets merges [existing, new], the episode indices in the
    merged dataset are:
      - existing episodes: 0 .. existing_total_episodes - 1
      - new episodes: existing_total_episodes .. existing_total_episodes + new_total_episodes - 1

    The replay/new split is determined by:
      - If existing has er_meta: replay = old replay + sampled from old new
      - If no er_meta: replay = sampled from all existing episodes
      - new = all episodes from the new dataset
    """
    rng = random.Random(seed)

    n_existing = existing_meta.total_episodes
    n_new = new_meta.total_episodes

    if existing_er_meta is not None:
        # Recursive case: existing is already a merged ER dataset
        old_replay_eps = existing_er_meta["replay_episodes"]
        old_new_eps = existing_er_meta["new_episodes"]

        # Sample from the old "new" episodes to become replay
        if sample_n_episodes is not None and sample_n_episodes < len(old_new_eps):
            sampled = sorted(rng.sample(old_new_eps, sample_n_episodes))
        else:
            sampled = old_new_eps

        replay_episodes = sorted(old_replay_eps + sampled)
    else:
        # Base case: existing is a raw dataset, sample episodes as replay
        all_existing = list(range(n_existing))
        if sample_n_episodes is not None and sample_n_episodes < n_existing:
            replay_episodes = sorted(rng.sample(all_existing, sample_n_episodes))
        else:
            replay_episodes = all_existing

    # New episodes get indices starting after existing
    new_episodes = list(range(n_existing, n_existing + n_new))

    return {
        "replay_episodes": replay_episodes,
        "new_episodes": new_episodes,
    }


def create_er_dataset(
    existing_repo_id: str,
    new_repo_id: str,
    output_repo_id: str,
    existing_root: str | None = None,
    new_root: str | None = None,
    output_root: str | None = None,
    sample_n_episodes: int | None = None,
    seed: int = 42,
):
    """Create an ER dataset by merging existing (replay) data with new data."""
    logging.basicConfig(level=logging.INFO)

    existing_root_path = Path(existing_root) if existing_root else None
    new_root_path = Path(new_root) if new_root else None
    output_root_path = Path(output_root) if output_root else None

    # Load metadata for both datasets
    existing_meta = LeRobotDatasetMetadata(
        existing_repo_id,
        root=existing_root_path / existing_repo_id if existing_root_path else None,
    )
    new_meta = LeRobotDatasetMetadata(
        new_repo_id,
        root=new_root_path / new_repo_id if new_root_path else None,
    )

    # Check for existing er_meta
    existing_er_meta = load_er_meta(existing_meta.root)

    # Compute episode split for the merged dataset
    er_meta = compute_er_meta(
        existing_meta, new_meta, existing_er_meta, sample_n_episodes, seed
    )

    logging.info(
        f"Merging: {len(er_meta['replay_episodes'])} replay episodes + "
        f"{len(er_meta['new_episodes'])} new episodes"
    )

    # Aggregate the two datasets
    # Pass actual dataset roots so aggregate_datasets resolves paths correctly
    roots = [existing_meta.root, new_meta.root]
    aggregate_datasets(
        repo_ids=[existing_repo_id, new_repo_id],
        aggr_repo_id=output_repo_id,
        roots=roots,
        aggr_root=output_root_path,
    )

    # Save er_meta.json in the output dataset
    if output_root_path:
        actual_output_root = output_root_path / output_repo_id
    else:
        actual_output_root = HF_LEROBOT_HOME / output_repo_id
    save_er_meta(actual_output_root, er_meta)

    logging.info("ER dataset creation complete.")
    logging.info(f"  Replay episodes: {er_meta['replay_episodes']}")
    logging.info(f"  New episodes: {er_meta['new_episodes']}")


def main():
    parser = argparse.ArgumentParser(description="Create an Experience Replay dataset")
    parser.add_argument("--existing_repo_id", type=str, required=True,
                        help="Repo ID of the existing/replay dataset")
    parser.add_argument("--existing_root", type=str, default=None,
                        help="Root path for the existing dataset")
    parser.add_argument("--new_repo_id", type=str, required=True,
                        help="Repo ID of the new dataset")
    parser.add_argument("--new_root", type=str, default=None,
                        help="Root path for the new dataset")
    parser.add_argument("--output_repo_id", type=str, required=True,
                        help="Repo ID for the output merged dataset")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root path for the output dataset")
    parser.add_argument("--sample_n_episodes", type=int, default=None,
                        help="Number of episodes to sample from existing 'new' episodes as replay")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for episode sampling")
    args = parser.parse_args()

    create_er_dataset(
        existing_repo_id=args.existing_repo_id,
        new_repo_id=args.new_repo_id,
        output_repo_id=args.output_repo_id,
        existing_root=args.existing_root,
        new_root=args.new_root,
        output_root=args.output_root,
        sample_n_episodes=args.sample_n_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
