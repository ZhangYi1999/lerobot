"""
Overlay all initial frames from both cameras of a LeRobot v3.0 dataset
to visualize the distribution of initial object positions.
"""

import argparse
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_first_frame_from_video(video_path: Path, pts_or_frame_idx: int = 0) -> np.ndarray:
    """Decode and return the first frame of a video file."""
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            return np.array(frame.to_image())
    raise ValueError(f"No frames found in {video_path}")


def get_first_frames(dataset_dir: Path, camera_key: str) -> list[np.ndarray]:
    """Return the first frame of each episode for a given camera."""
    data_dir = dataset_dir / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parquet_files])

    # One row per episode where frame_index == 0
    first_rows = df[df["frame_index"] == 0].sort_values("episode_index")

    video_dir = dataset_dir / "videos" / camera_key / "chunk-000"
    video_files = sorted(video_dir.glob("*.mp4"))

    # Build a map from global frame index → video file + local position
    frame_offsets = []  # (start_global_idx, video_path)
    cumulative = 0
    for vf in video_files:
        frame_offsets.append((cumulative, vf))
        with av.open(str(vf)) as c:
            cumulative += c.streams.video[0].frames

    def find_video_and_local_idx(global_idx: int):
        for i, (start, vf) in enumerate(frame_offsets):
            end = frame_offsets[i + 1][0] if i + 1 < len(frame_offsets) else cumulative
            if start <= global_idx < end:
                return vf, global_idx - start
        raise ValueError(f"Global index {global_idx} out of range")

    frames = []
    for _, row in first_rows.iterrows():
        video_path, local_idx = find_video_and_local_idx(int(row["index"]))
        with av.open(str(video_path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i == local_idx:
                    frames.append(np.array(frame.to_image()))
                    break
    return frames


def make_overlay(frames: list[np.ndarray], alpha: float = 0.15) -> np.ndarray:
    """Blend all frames into a single image via mean."""
    stack = np.stack(frames, axis=0).astype(np.float32)
    return np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="continuallearning/real_0_put_bowl_filtered")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache/huggingface/lerobot"))
    parser.add_argument("--output", default="initial_frames_overlay.png")
    args = parser.parse_args()

    dataset_dir = Path(args.cache_dir) / args.dataset.replace("/", "/")

    cameras = ["observation.images.primary", "observation.images.wrist"]
    camera_labels = ["Third-person camera", "Wrist camera"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"Initial frame overlay — {args.dataset}\n"
        f"({len(pd.concat([pd.read_parquet(f) for f in sorted((dataset_dir / 'data' / 'chunk-000').glob('*.parquet'))])['episode_index'].unique())} episodes)",
        fontsize=13,
    )

    for ax, camera_key, label in zip(axes, cameras, camera_labels):
        print(f"Extracting first frames for {camera_key}…")
        frames = get_first_frames(dataset_dir, camera_key)
        print(f"  Got {len(frames)} frames")
        overlay = make_overlay(frames)
        ax.imshow(overlay)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(args.output)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
