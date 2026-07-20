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
"""Train a policy with DAgger (Dataset Aggregation).

DAgger is an interactive imitation-learning method: the *student* policy acts in the environment, an
*expert* policy relabels every visited state, and the aggregated expert-labeled data is used to keep
training the student. A "mixed rollout" executes the expert's action with probability ``beta`` (which
decays over iterations), so the visited-state distribution shifts smoothly from expert- to
student-induced. Evaluation always uses the pure student.

This mirrors the DAgger recipe from RLinf (expert-model relabeling + beta scheduling + replay/dataset
aggregation), implemented on top of LeRobot's gym-env rollout and policy-training building blocks.

Requires: pip install 'lerobot[evaluation]' plus the policy extra (e.g. lerobot[pi]) and the
          environment extra (e.g. lerobot[libero]).

Usage example (student and expert are both LeRobot policy checkpoints):

```
lerobot-dagger \
    --env.type=libero \
    --policy.path=outputs/train/student/checkpoints/last/pretrained_model \
    --expert_policy.path=lerobot/pi0_libero \
    --dagger.n_iterations=20 \
    --dagger.episodes_per_iteration=10 \
    --dagger.steps_per_iteration=500 \
    --dagger.init_beta=1.0 \
    --dagger.beta_decay=0.9
```
"""

import json
import logging
import random
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.dagger import DAggerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import (
    _build_raw_frame,
    _env_features_to_dataset_features,
    eval_policy_all,
)
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import has_method, init_logging


class BetaScheduler:
    """Schedules the expert-action probability ``beta`` across DAgger iterations.

    ``beta`` starts at ``init_beta`` and decays toward ``beta_min``:
    - ``exponential``: ``beta = max(beta_min, init_beta * beta_decay ** iteration)``
    - ``linear``: ``beta = max(beta_min, init_beta - (1 - beta_decay) * iteration)``
    - ``constant``: ``beta = init_beta``
    """

    def __init__(
        self,
        init_beta: float = 1.0,
        schedule: str = "exponential",
        decay: float = 0.99,
        min_beta: float = 0.05,
    ):
        self.init_beta = init_beta
        self.schedule = schedule
        self.decay = decay
        self.min_beta = min_beta

    def value(self, iteration: int) -> float:
        if self.schedule == "constant":
            beta = self.init_beta
        elif self.schedule == "linear":
            beta = self.init_beta - (1.0 - self.decay) * iteration
        else:  # exponential
            beta = self.init_beta * (self.decay**iteration)
        return float(max(self.min_beta, min(self.init_beta, beta)))


def _flatten_envs(envs: dict) -> list:
    """Flatten the nested {suite: {task_id: vec_env}} structure returned by `make_env`."""
    return [vec for group in envs.values() for vec in group.values()]


def _extract_successes(info: dict, num_envs: int) -> list[bool]:
    """Read per-env success flags from a gym `info` dict (mirrors lerobot_eval.rollout)."""
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            is_success = final_info.get("is_success", [False] * num_envs)
            return is_success.tolist() if hasattr(is_success, "tolist") else [bool(is_success)] * num_envs
        successes = []
        for item in final_info:
            if isinstance(item, dict) and "is_success" in item:
                successes.append(bool(item["is_success"]))
            else:
                successes.append(False)
        return successes
    if "is_success" in info:
        is_success = info["is_success"]
        return is_success.tolist() if hasattr(is_success, "tolist") else [bool(is_success)] * num_envs
    return [False] * num_envs


@torch.no_grad()
def collect_rollout_episodes(
    env,
    student: PreTrainedPolicy,
    expert: PreTrainedPolicy,
    student_env_pre,
    student_env_post,
    student_pre,
    student_post,
    expert_env_pre,
    expert_env_post,
    expert_pre,
    expert_post,
    env_features: dict,
    beta: float,
    only_save_expert: bool,
    seeds: list[int] | None = None,
) -> list[dict]:
    """Run one mixed rollout over a vectorized env, relabeling each visited state with the expert.

    Executed action = expert action with probability ``beta`` (per env, per step), else student action.
    The stored supervision target is always the expert action. Returns one dict per env:
    ``{"frames": [...], "success": bool}``.
    """
    student.reset()
    expert.reset()
    observation, info = env.reset(seed=seeds)

    num_envs = env.num_envs
    frame_buffers: list[list[dict]] = [[] for _ in range(num_envs)]
    episode_success = [False] * num_envs
    done = np.array([False] * num_envs)

    raw_observation = deepcopy(observation)
    max_steps = env.call("_max_episode_steps")[0]
    check_env_attributes_and_types(env)

    step = 0
    while not np.all(done) and step < max_steps:
        base_obs = preprocess_observation(observation)
        try:
            tasks = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            try:
                tasks = list(env.call("task"))
            except (AttributeError, NotImplementedError):
                tasks = [""] * num_envs

        # --- Student forward pass (drives exploration) ---
        student_obs = deepcopy(base_obs)
        student_obs["task"] = list(tasks)
        student_obs = student_pre(student_env_pre(student_obs))
        student_action = student.select_action(student_obs)
        student_action = student_env_post(student_post(student_action))
        student_action_np = student_action.to("cpu").numpy()

        # --- Expert forward pass (produces supervision labels) ---
        expert_obs = deepcopy(base_obs)
        expert_obs["task"] = list(tasks)
        expert_obs = expert_pre(expert_env_pre(expert_obs))
        expert_action = expert.select_action(expert_obs)
        expert_action = expert_env_post(expert_post(expert_action))
        expert_action_np = expert_action.to("cpu").numpy()

        # --- Mixed execution: expert with prob beta, else student ---
        use_expert = np.array([random.random() < beta for _ in range(num_envs)])
        executed = np.where(use_expert[:, None], expert_action_np, student_action_np)
        assert executed.ndim == 2, "Action dimensions should be (batch, action_dim)"

        observation, reward, terminated, truncated, info = env.step(executed)
        successes = _extract_successes(info, num_envs)

        prev_done = done.copy()
        for env_idx in range(num_envs):
            if prev_done[env_idx]:
                continue
            if only_save_expert and not use_expert[env_idx]:
                continue
            frame = _build_raw_frame(
                raw_observation,
                env_idx,
                expert_action_np[env_idx],  # supervision target is always the expert action
                reward[env_idx],
                successes[env_idx],
                bool(terminated[env_idx] | truncated[env_idx]),
                tasks[env_idx] if isinstance(tasks[env_idx], str) else "",
                env_features,
            )
            frame_buffers[env_idx].append(frame)
            if successes[env_idx]:
                episode_success[env_idx] = True

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)
        raw_observation = deepcopy(observation)
        step += 1

    return [
        {"frames": frame_buffers[i], "success": episode_success[i]}
        for i in range(num_envs)
        if len(frame_buffers[i]) > 0
    ]


def append_episodes(dataset: LeRobotDataset, episodes: list[dict], only_success: bool) -> tuple[int, int]:
    """Write collected episodes into the aggregated dataset. Returns (n_episodes, n_frames) added."""
    n_eps = 0
    n_frames = 0
    for episode in episodes:
        if only_success and not episode["success"]:
            continue
        frames = episode["frames"]
        if len(frames) == 0:
            continue
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        n_eps += 1
        n_frames += len(frames)
    return n_eps, n_frames


def train_student(
    policy: PreTrainedPolicy,
    preprocessor,
    dataset: LeRobotDataset,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    cfg: DAggerPipelineConfig,
    device: torch.device,
) -> float:
    """Run supervised BC training on the aggregated dataset for `steps_per_iteration` steps."""
    dcfg = cfg.dagger
    use_amp = cfg.policy.use_amp
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=dcfg.num_workers,
        batch_size=dcfg.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )
    camera_keys = dataset.meta.camera_keys

    policy.train()
    loss_sum = 0.0
    n = 0
    dl_iter = iter(dataloader)
    for train_step in range(dcfg.steps_per_iteration):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        for cam_key in camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)

        with torch.autocast(device_type=device.type) if use_amp else nullcontext():
            loss, _ = policy.forward(batch)

        loss.backward()
        if dcfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), dcfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if has_method(policy, "update"):
            policy.update()

        loss_sum += loss.item()
        n += 1
        if dcfg.log_freq > 0 and (train_step + 1) % dcfg.log_freq == 0:
            logging.info(f"  train step {train_step + 1}/{dcfg.steps_per_iteration}: loss={loss.item():.4f}")

    return loss_sum / max(n, 1)


def save_student_checkpoint(policy, preprocessor, postprocessor, checkpoint_dir: Path) -> None:
    """Save the student so it can be reloaded with `--policy.path=<dir>/pretrained_model`."""
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(pretrained_dir)
    policy.config.save_pretrained(pretrained_dir)
    preprocessor.save_pretrained(pretrained_dir)
    postprocessor.save_pretrained(pretrained_dir)


def dagger_main(cfg: DAggerPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError(
            "DAgger requires a pretrained student (`--policy.path=...`) so its normalization stats and "
            "processors are available for both rollout and training."
        )

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Build environments (shared for collection and evaluation) ----
    logging.info(f"Making environment (rollout_batch_size={cfg.dagger.rollout_batch_size}).")
    envs = make_env(
        cfg.env,
        n_envs=cfg.dagger.rollout_batch_size,
        use_async_envs=cfg.dagger.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    vec_envs = _flatten_envs(envs)

    # ---- Build student policy + processors + optimizer ----
    logging.info("Making student policy.")
    student = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    student_pre, student_post = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(student.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    student_env_pre, student_env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    optimizer_cfg = cfg.policy.get_optimizer_preset()
    optimizer = optimizer_cfg.build(student.get_optim_params())
    scheduler_cfg = cfg.policy.get_scheduler_preset()
    total_steps = cfg.dagger.n_iterations * cfg.dagger.steps_per_iteration
    lr_scheduler = scheduler_cfg.build(optimizer, total_steps) if scheduler_cfg is not None else None

    # ---- Build (frozen) expert policy + processors ----
    logging.info("Making expert policy.")
    expert = make_policy(cfg=cfg.expert_policy, env_cfg=cfg.env, rename_map=cfg.expert_rename_map)
    expert.eval()
    for p in expert.parameters():
        p.requires_grad_(False)
    expert_pre, expert_post = make_pre_post_processors(
        policy_cfg=cfg.expert_policy,
        pretrained_path=cfg.expert_policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(expert.config.device)},
            "rename_observations_processor": {"rename_map": cfg.expert_rename_map},
        },
    )
    expert_env_pre, expert_env_post = make_env_pre_post_processors(
        env_cfg=cfg.env, policy_cfg=cfg.expert_policy
    )

    # ---- Create the aggregated dataset ----
    features = _env_features_to_dataset_features(cfg.env.features)
    fps = vec_envs[0].unwrapped.metadata.get("render_fps", 30)
    dataset = LeRobotDataset.create(
        repo_id=cfg.dataset_repo_id,
        fps=fps,
        features=features,
        root=str(cfg.dataset_root),
        use_videos=True,
    )

    beta_scheduler = BetaScheduler(
        init_beta=cfg.dagger.init_beta,
        schedule=cfg.dagger.beta_schedule,
        decay=cfg.dagger.beta_decay,
        min_beta=cfg.dagger.beta_min,
    )

    metrics_log: list[dict] = []

    # ---- Main DAgger loop ----
    for iteration in range(cfg.dagger.n_iterations):
        beta = beta_scheduler.value(iteration)
        logging.info(
            colored(f"\n=== DAgger iteration {iteration + 1}/{cfg.dagger.n_iterations} ===", "green")
            + f" beta={beta:.3f}"
        )

        # --- Collection: mixed rollout with expert relabeling ---
        student.eval()
        collected_eps = 0
        collected_frames = 0
        env_cycle = 0
        seed_base = (cfg.seed or 0) + iteration * 100_000
        while collected_eps < cfg.dagger.episodes_per_iteration:
            vec = vec_envs[env_cycle % len(vec_envs)]
            env_cycle += 1
            seeds = [seed_base + env_cycle * 1000 + i for i in range(vec.num_envs)]
            episodes = collect_rollout_episodes(
                vec,
                student,
                expert,
                student_env_pre,
                student_env_post,
                student_pre,
                student_post,
                expert_env_pre,
                expert_env_post,
                expert_pre,
                expert_post,
                env_features=dataset.features,
                beta=beta,
                only_save_expert=cfg.dagger.only_save_expert,
                seeds=seeds,
            )
            n_eps, n_frames = append_episodes(dataset, episodes, cfg.dagger.only_success)
            collected_eps += n_eps
            collected_frames += n_frames

        logging.info(
            f"Collected {collected_eps} episodes / {collected_frames} frames. "
            f"Aggregated dataset now has {dataset.num_episodes} episodes / {dataset.num_frames} frames."
        )

        # --- Training on aggregated data ---
        avg_loss = train_student(student, student_pre, dataset, optimizer, lr_scheduler, cfg, device)
        logging.info(f"Iteration {iteration + 1} avg actor loss: {avg_loss:.4f}")

        iter_metrics = {
            "iteration": iteration + 1,
            "beta": beta,
            "actor_loss": avg_loss,
            "buffer_num_episodes": dataset.num_episodes,
            "buffer_num_frames": dataset.num_frames,
        }

        # --- Evaluation with the pure student ---
        if cfg.dagger.eval_freq > 0 and (iteration + 1) % cfg.dagger.eval_freq == 0:
            student.eval()
            with torch.no_grad():
                eval_info = eval_policy_all(
                    envs=envs,
                    policy=student,
                    env_preprocessor=student_env_pre,
                    env_postprocessor=student_env_post,
                    preprocessor=student_pre,
                    postprocessor=student_post,
                    n_episodes=cfg.dagger.eval_n_episodes,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )
            success = eval_info["overall"].get("pc_success")
            iter_metrics["success_once"] = success
            logging.info(colored(f"Iteration {iteration + 1} student success rate: {success}", "yellow"))

        metrics_log.append(iter_metrics)

        # --- Checkpoint the student ---
        checkpoint_dir = Path(cfg.output_dir) / "checkpoints" / f"{iteration + 1:06d}"
        save_student_checkpoint(student, student_pre, student_post, checkpoint_dir)
        logging.info(f"Saved student checkpoint to {checkpoint_dir}")

    # Save a `last` pointer + metrics summary.
    save_student_checkpoint(student, student_pre, student_post, Path(cfg.output_dir) / "checkpoints" / "last")
    with open(Path(cfg.output_dir) / "dagger_metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    close_envs(envs)
    logging.info("End of DAgger training")


@parser.wrap()
def main(cfg: DAggerPipelineConfig):
    init_logging()
    register_third_party_plugins()
    dagger_main(cfg)


if __name__ == "__main__":
    main()
