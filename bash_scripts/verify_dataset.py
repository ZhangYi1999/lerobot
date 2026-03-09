#!/usr/bin/env python3
"""Download lerobot/libero_10_subtask and verify episode-task mapping."""

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

print("Loading dataset (will download if needed)...")
ds = LeRobotDataset("lerobot/libero_10_subtask")

print(f"\nDataset info:")
print(f"  Total episodes: {ds.meta.total_episodes}")
print(f"  Total frames:   {ds.meta.total_frames}")
print(f"  FPS:            {ds.meta.fps}")

# Build episode -> task_index mapping
task_map: dict[int, list[int]] = {}
for ep_idx in range(ds.meta.total_episodes):
    # Get first frame of each episode to read task_index
    ep_data = ds.meta.episodes[ep_idx]
    task_idx = ep_data.get("task_index", ep_data.get("tasks", {}).get("task_index"))
    if task_idx is None:
        # Fallback: read from actual data
        from datasets import load_dataset
        break

    task_map.setdefault(task_idx, []).append(ep_idx)

# If meta didn't have task_index, read from parquet
if not task_map:
    print("\nReading task_index from parquet data...")
    import pyarrow.parquet as pq
    from pathlib import Path
    import huggingface_hub

    # Find the data files
    data_dir = ds.root
    parquet_files = sorted(Path(data_dir).rglob("*.parquet"))

    for pf in parquet_files:
        table = pq.read_table(pf, columns=["episode_index", "task_index"])
        ep_col = table.column("episode_index").to_pylist()
        task_col = table.column("task_index").to_pylist()
        for ep, task in zip(ep_col, task_col):
            if ep not in [e for eps in task_map.values() for e in eps]:
                task_map.setdefault(task, []).append(ep)

    # Deduplicate
    for k in task_map:
        task_map[k] = sorted(set(task_map[k]))

print("\n" + "=" * 60)
print("TASK → EPISODE MAPPING")
print("=" * 60)
for task_idx in sorted(task_map.keys()):
    episodes = sorted(task_map[task_idx])
    print(f"  Task {task_idx:2d}: episodes {episodes[0]:3d}..{episodes[-1]:3d}  ({len(episodes)} episodes)")

# Verify assumptions
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
assumed_correct = True
for task_idx in sorted(task_map.keys()):
    episodes = sorted(task_map[task_idx])
    expected = list(range(task_idx * 50, task_idx * 50 + 50))
    if episodes == expected:
        print(f"  Task {task_idx}: ✓ matches assumed mapping [task*50 .. task*50+49]")
    else:
        print(f"  Task {task_idx}: ✗ MISMATCH! Expected {expected[0]}..{expected[-1]}, got {episodes[0]}..{episodes[-1]}")
        assumed_correct = False

if assumed_correct:
    print("\n✓ All task-episode mappings match the assumed sequential layout.")
    print("  common.sh get_episode_list() is correct.")
else:
    print("\n✗ Mapping mismatch! Update common.sh get_episode_list() accordingly.")
    print("\nActual mapping for common.sh:")
    print("TASK_EPISODES=(")
    for task_idx in sorted(task_map.keys()):
        episodes = sorted(task_map[task_idx])
        ep_str = ",".join(str(e) for e in episodes)
        print(f'    "{ep_str}"  # task {task_idx}')
    print(")")
