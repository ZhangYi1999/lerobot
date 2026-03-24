"""
Consolidate a LeRobot dataset from many small files into fewer large files.

Reduces CPU RAM during training by cutting the number of torchcodec
VideoDecoder objects each DataLoader worker must keep alive. With 100
episodes x 2 cameras = 200 MP4 files and num_workers=8, decoder caches
alone consume ~27 GB. Merging all episodes into 1 MP4 per camera drops
this to ~0.3 GB.

Usage
-----
    python -m lerobot.scripts.consolidate_dataset \\
        --dataset ~/data/real_0_put_bowl_filtered \\
        --output  /tmp/consolidated \\
        --video-file-size-mb 99999 \\
        --data-file-size-mb  99999 \\
        --push-to-hub \\
        --hub-repo-id continuallearning/real_0_put_bowl_filtered_consolidated
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    STATS_PATH,
    update_chunk_file_indices,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_all_episodes(root: Path) -> pd.DataFrame:
    """Load and concatenate all shard parquets under meta/episodes/."""
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No episode parquets found under {root}/meta/episodes/"
        )
    return (
        pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )


def concat_videos_ffmpeg(src_paths: list[Path], dst: Path) -> None:
    """Concatenate src MP4s into dst via ffmpeg stream-copy (lossless)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        for p in src_paths:
            f.write(f"file '{p.resolve()}'\n")
        concat_list = f.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                str(dst),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        Path(concat_list).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Video consolidation
# ---------------------------------------------------------------------------


def consolidate_videos(
    root: Path,
    out_root: Path,
    eps_df: pd.DataFrame,
    video_keys: list[str],
    max_size_mb: float,
    chunks_size: int,
) -> None:
    """Merge per-episode MP4s into fewer larger files; mutates eps_df."""
    n_episodes = len(eps_df)

    for vid_key in video_keys:
        log.info(f"  Consolidating video key: {vid_key}")
        chunk_idx = file_idx = 0
        cum_size = 0.0
        cum_dur = 0.0
        group_src: list[Path] = []
        group_ep_indices: list[int] = []

        def _flush(c_idx: int, f_idx: int, srcs: list[Path]) -> None:
            dst = out_root / DEFAULT_VIDEO_PATH.format(
                video_key=vid_key,
                chunk_index=c_idx,
                file_index=f_idx,
            )
            log.info(
                f"    Writing chunk-{c_idx:03d}/file-{f_idx:03d}.mp4"
                f"  ({len(srcs)} episodes)"
            )
            concat_videos_ffmpeg(srcs, dst)

        for row_i, ep_data in eps_df.iterrows():
            ep_idx: int = int(ep_data["episode_index"])
            c_key = f"videos/{vid_key}/chunk_index"
            f_key = f"videos/{vid_key}/file_index"
            ts_from = f"videos/{vid_key}/from_timestamp"
            ts_to = f"videos/{vid_key}/to_timestamp"

            src = root / DEFAULT_VIDEO_PATH.format(
                video_key=vid_key,
                chunk_index=int(ep_data[c_key]),
                file_index=int(ep_data[f_key]),
            )
            ep_dur = float(ep_data[ts_to]) - float(ep_data[ts_from])
            ep_size = src.stat().st_size / 1024**2

            # Flush current group if this episode would exceed limit
            if max_size_mb > 0 and group_src and cum_size + ep_size > max_size_mb:
                _flush(chunk_idx, file_idx, group_src)
                chunk_idx, file_idx = update_chunk_file_indices(
                    chunk_idx, file_idx, chunks_size
                )
                cum_size = 0.0
                cum_dur = 0.0
                group_src = []

            eps_df.at[row_i, c_key] = chunk_idx
            eps_df.at[row_i, f_key] = file_idx
            eps_df.at[row_i, ts_from] = cum_dur
            eps_df.at[row_i, ts_to] = cum_dur + ep_dur

            group_src.append(src)
            group_ep_indices.append(ep_idx)
            cum_size += ep_size
            cum_dur += ep_dur

        if group_src:
            _flush(chunk_idx, file_idx, group_src)

        dst_count = file_idx + 1 + chunk_idx * chunks_size
        log.info(f"  {vid_key}: {n_episodes} → {dst_count} file(s)")


# ---------------------------------------------------------------------------
# Data parquet consolidation
# ---------------------------------------------------------------------------


def consolidate_data(
    root: Path,
    out_root: Path,
    eps_df: pd.DataFrame,
    max_size_mb: float,
    chunks_size: int,
) -> None:
    """Merge per-episode data parquets; mutates eps_df."""
    log.info("  Consolidating data parquets")

    all_data_paths = sorted((root / "data").rglob("*.parquet"))
    all_data = (
        pd.concat([pd.read_parquet(p) for p in all_data_paths])
        .drop_duplicates(subset=["index"])
        .sort_values("index")
        .reset_index(drop=True)
    )
    n_dup = sum(len(pd.read_parquet(p)) for p in all_data_paths) - len(all_data)
    if n_dup > 0:
        log.warning(f"  Dropped {n_dup} duplicate frame rows (same 'index' value)")

    chunk_idx = file_idx = 0
    cum_size_mb = 0.0
    pending: list[pd.DataFrame] = []

    def _flush(c_idx: int, f_idx: int, rows: list[pd.DataFrame]) -> None:
        dst = out_root / DEFAULT_DATA_PATH.format(
            chunk_index=c_idx, file_index=f_idx
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        merged = pd.concat(rows)
        merged.to_parquet(dst, index=False)
        log.info(
            f"    data chunk-{c_idx:03d}/file-{f_idx:03d}.parquet"
            f"  ({len(merged)} frames)"
        )

    for row_i, ep_data in eps_df.iterrows():
        ep_idx: int = int(ep_data["episode_index"])
        from_idx = int(ep_data["dataset_from_index"])
        to_idx = int(ep_data["dataset_to_index"])
        ep_rows = all_data[(all_data["index"] >= from_idx) & (all_data["index"] < to_idx)]
        ep_size_mb = ep_rows.memory_usage(deep=True).sum() / 1024**2

        if max_size_mb > 0 and pending and cum_size_mb + ep_size_mb > max_size_mb:
            _flush(chunk_idx, file_idx, pending)
            chunk_idx, file_idx = update_chunk_file_indices(
                chunk_idx, file_idx, chunks_size
            )
            cum_size_mb = 0.0
            pending = []

        eps_df.at[row_i, "data/chunk_index"] = chunk_idx
        eps_df.at[row_i, "data/file_index"] = file_idx

        pending.append(ep_rows)
        cum_size_mb += ep_size_mb

    if pending:
        _flush(chunk_idx, file_idx, pending)

    dst_count = file_idx + 1 + chunk_idx * chunks_size
    log.info(f"  data: {len(all_data_paths)} → {dst_count} file(s)")


# ---------------------------------------------------------------------------
# Write metadata
# ---------------------------------------------------------------------------


def write_episodes_meta(out_root: Path, eps_df: pd.DataFrame) -> None:
    """Write consolidated episodes metadata as a single parquet shard."""
    dst = out_root / DEFAULT_EPISODES_PATH.format(
        chunk_index=0, file_index=0
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = eps_df.copy()
    out["meta/episodes/chunk_index"] = 0
    out["meta/episodes/file_index"] = 0
    out.to_parquet(dst, index=False)
    log.info(f"  Episodes metadata → {dst}")


def copy_unchanged_files(root: Path, out_root: Path) -> None:
    """Copy stats.json, tasks.parquet, subtasks.parquet unchanged."""
    for rel in [STATS_PATH, "meta/tasks.parquet", "meta/subtasks.parquet"]:
        src = root / rel
        if src.exists():
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_info_json(
    root: Path,
    out_root: Path,
    data_size_mb: float,
    video_size_mb: float,
) -> None:
    info = json.loads((root / INFO_PATH).read_text())
    if data_size_mb > 0:
        info["data_files_size_in_mb"] = data_size_mb
    if video_size_mb > 0:
        info["video_files_size_in_mb"] = video_size_mb
    dst = out_root / INFO_PATH
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(info, indent=4))


def push_to_hub(output_path: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    log.info(f"Creating/verifying Hub repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    log.info(f"Uploading {output_path} → {repo_id} ...")
    api.upload_folder(
        folder_path=str(output_path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    log.info(f"Pushed to https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate LeRobot dataset files."
    )
    parser.add_argument(
        "--dataset", required=True, type=str,
        help=(
            "Source dataset: local path OR HuggingFace repo_id "
            "(e.g. continuallearning/real_0_put_bowl_filtered). "
            "Repo IDs are resolved to $HF_LEROBOT_HOME/<repo_id>."
        ),
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output dataset root (must not equal --dataset).",
    )
    parser.add_argument(
        "--video-file-size-mb", type=float, default=0.0,
        help=(
            "Max MB per output video file. "
            "0=keep 1/episode (skip consolidation). "
            "99999=merge all into one file."
        ),
    )
    parser.add_argument(
        "--data-file-size-mb", type=float, default=0.0,
        help=(
            "Max MB per output data parquet. "
            "0=keep current layout unchanged. "
            "99999=merge all into one file."
        ),
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push the output dataset to HuggingFace Hub.",
    )
    parser.add_argument(
        "--hub-repo-id", type=str, default=None,
        help="Hub repo_id (required with --push-to-hub).",
    )
    args = parser.parse_args()

    # Resolve dataset: local path or HuggingFace repo_id
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        root = dataset_path.resolve()
    else:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        root = HF_LEROBOT_HOME / args.dataset
        if not root.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{dataset_path}' or '{root}'. "
                f"Pass a valid local path or a repo_id cached under "
                f"$HF_LEROBOT_HOME."
            )
        log.info(f"Resolved repo_id '{args.dataset}' → {root}")

    out_root: Path = args.output.resolve()

    if out_root == root:
        raise ValueError(
            "--output must differ from --dataset to avoid in-place modification."
        )
    if args.push_to_hub and not args.hub_repo_id:
        raise ValueError("--hub-repo-id is required when using --push-to-hub.")

    if out_root.exists():
        log.warning(f"Output directory already exists: {out_root}  (will overwrite)")

    log.info(f"Source:  {root}")
    log.info(f"Output:  {out_root}")

    info = json.loads((root / INFO_PATH).read_text())
    chunks_size: int = info.get("chunks_size", 1000)
    video_keys: list[str] = [
        k for k, v in info.get("features", {}).items()
        if v.get("dtype") == "video"
    ]

    log.info(
        f"Episodes: {info.get('total_episodes')}  |  "
        f"Video keys: {video_keys}"
    )
    log.info(
        f"chunks_size={chunks_size}  "
        f"video-file-size-mb={args.video_file_size_mb}  "
        f"data-file-size-mb={args.data_file_size_mb}"
    )

    eps_df = load_all_episodes(root)
    log.info(f"Loaded {len(eps_df)} episode records.")

    # --- Video consolidation ---
    if args.video_file_size_mb != 0 and video_keys:
        log.info("=== Video consolidation ===")
        consolidate_videos(
            root, out_root, eps_df,
            video_keys, args.video_file_size_mb, chunks_size,
        )
    else:
        log.info(
            "Skipping video consolidation (--video-file-size-mb=0). "
            "Copying video files as-is."
        )
        for vid_key in video_keys:
            src_dir = root / "videos" / vid_key
            dst_dir = out_root / "videos" / vid_key
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    # --- Data parquet consolidation ---
    if args.data_file_size_mb != 0:
        log.info("=== Data consolidation ===")
        consolidate_data(
            root, out_root, eps_df,
            args.data_file_size_mb, chunks_size,
        )
    else:
        log.info(
            "Skipping data consolidation (--data-file-size-mb=0). "
            "Copying data files as-is."
        )
        src_dir = root / "data"
        dst_dir = out_root / "data"
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    # --- Write metadata ---
    log.info("=== Writing metadata ===")
    write_episodes_meta(out_root, eps_df)
    copy_unchanged_files(root, out_root)
    write_info_json(
        root, out_root, args.data_file_size_mb, args.video_file_size_mb
    )

    # --- Summary ---
    n_mp4_before = (
        sum(1 for _ in (root / "videos").rglob("*.mp4"))
        if (root / "videos").exists() else 0
    )
    n_mp4_after = (
        sum(1 for _ in (out_root / "videos").rglob("*.mp4"))
        if (out_root / "videos").exists() else 0
    )
    n_data_before = (
        sum(1 for _ in (root / "data").rglob("*.parquet"))
        if (root / "data").exists() else 0
    )
    n_data_after = (
        sum(1 for _ in (out_root / "data").rglob("*.parquet"))
        if (out_root / "data").exists() else 0
    )
    log.info(
        f"Done.  "
        f"MP4: {n_mp4_before} → {n_mp4_after}  |  "
        f"data parquets: {n_data_before} → {n_data_after}"
    )

    # --- Push to Hub ---
    if args.push_to_hub:
        log.info("=== Pushing to HuggingFace Hub ===")
        push_to_hub(out_root, args.hub_repo_id)


if __name__ == "__main__":
    main()
