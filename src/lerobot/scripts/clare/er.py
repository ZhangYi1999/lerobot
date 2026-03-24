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
import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.clare.create_er_dataset import ER_META_FILENAME
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)

# Controls whether to reuse normalization stats from pretrained checkpoint.
REUSE_PRETRAINED_NORMALIZATION: bool = os.environ.get("REUSE_PRETRAINED_NORMALIZATION", "true").lower() != "false"

# When set, load normalizer stats from this JSON file (same format as meta/stats.json).
NORM_STATS_FILE: str | None = os.environ.get("NORM_STATS_FILE")


class ERBatchSampler:
    """Batch sampler that ensures each batch is 50% replay and 50% new data.

    Uses episode IDs from er_meta.json to split frames into replay and new pools,
    then yields batches with half from each pool.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        replay_episodes: list[int],
        new_episodes: list[int],
        batch_size: int,
        drop_n_last_frames: int = 0,
        shuffle: bool = True,
    ):
        assert batch_size % 2 == 0, f"batch_size must be even, got {batch_size}"
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.shuffle = shuffle

        replay_set = set(replay_episodes)
        new_set = set(new_episodes)

        self.replay_indices = []
        self.new_indices = []

        for ep_idx, (start, end) in enumerate(zip(dataset_from_indices, dataset_to_indices, strict=True)):
            frame_indices = list(range(start, end - drop_n_last_frames))
            if ep_idx in replay_set:
                self.replay_indices.extend(frame_indices)
            elif ep_idx in new_set:
                self.new_indices.extend(frame_indices)

        if not self.replay_indices:
            raise ValueError("No replay frames found. Check er_meta.json replay_episodes.")
        if not self.new_indices:
            raise ValueError("No new frames found. Check er_meta.json new_episodes.")

        logging.info(
            f"ERBatchSampler: {len(self.replay_indices)} replay frames, "
            f"{len(self.new_indices)} new frames, batch_size={batch_size}"
        )

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            replay_perm = torch.randperm(len(self.replay_indices)).tolist()
            new_perm = torch.randperm(len(self.new_indices)).tolist()
        else:
            replay_perm = list(range(len(self.replay_indices)))
            new_perm = list(range(len(self.new_indices)))

        # Cycle the smaller pool to match the larger
        n_batches = max(
            len(self.replay_indices) // self.half_batch,
            len(self.new_indices) // self.half_batch,
        )

        replay_idx = 0
        new_idx = 0
        for _ in range(n_batches):
            batch = []
            # Replay half
            for _ in range(self.half_batch):
                if replay_idx >= len(replay_perm):
                    # Reshuffle and cycle
                    replay_perm = torch.randperm(len(self.replay_indices)).tolist() if self.shuffle else list(range(len(self.replay_indices)))
                    replay_idx = 0
                batch.append(self.replay_indices[replay_perm[replay_idx]])
                replay_idx += 1
            # New half
            for _ in range(self.half_batch):
                if new_idx >= len(new_perm):
                    new_perm = torch.randperm(len(self.new_indices)).tolist() if self.shuffle else list(range(len(self.new_indices)))
                    new_idx = 0
                batch.append(self.new_indices[new_perm[new_idx]])
                new_idx += 1
            yield batch

    def __len__(self) -> int:
        return max(
            len(self.replay_indices) // self.half_batch,
            len(self.new_indices) // self.half_batch,
        )


def load_er_meta(dataset_root) -> dict:
    """Load er_meta.json from dataset root."""
    meta_path = dataset_root / ER_META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"er_meta.json not found at {meta_path}. "
            "Run create_er_dataset.py first to create the merged ER dataset."
        )
    with open(meta_path) as f:
        return json.load(f)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    force_cpu = cfg.policy.device == "cpu"
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        cpu=force_cpu,
    )

    init_logging(accelerator=accelerator)
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Load ER episode split metadata
    er_meta = load_er_meta(dataset.meta.root)
    logging.info(
        f"ER dataset: {len(er_meta['replay_episodes'])} replay episodes, "
        f"{len(er_meta['new_episodes'])} new episodes"
    )

    # Create evaluation environment
    eval_env = None
    env_preprocessor = env_postprocessor = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # Create preprocessor/postprocessor
    _norm_stats = None
    if NORM_STATS_FILE:
        logging.info(f"Loading normalizer stats from NORM_STATS_FILE: {NORM_STATS_FILE}")
        with open(NORM_STATS_FILE) as f:
            _norm_stats = json.load(f)
    elif not (cfg.policy.pretrained_path and REUSE_PRETRAINED_NORMALIZATION):
        _norm_stats = dataset.meta.stats

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        if _norm_stats is not None:
            processor_kwargs["dataset_stats"] = _norm_stats
    if cfg.policy.pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        if _norm_stats is not None:
            preprocessor_overrides["normalizer_processor"] = {
                "stats": _norm_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            }
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides

        postprocessor_overrides = {}
        if _norm_stats is not None:
            postprocessor_overrides["unnormalizer_processor"] = {
                "stats": _norm_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        if postprocessor_overrides:
            postprocessor_kwargs["postprocessor_overrides"] = postprocessor_overrides
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Create ER batch sampler: each batch is 50% replay + 50% new
    drop_n_last = getattr(cfg.policy, "drop_n_last_frames", 0)
    batch_sampler = ERBatchSampler(
        dataset_from_indices=dataset.meta.episodes["dataset_from_index"],
        dataset_to_indices=dataset.meta.episodes["dataset_to_index"],
        replay_episodes=er_meta["replay_episodes"],
        new_episodes=er_meta["new_episodes"],
        batch_size=cfg.batch_size,
        drop_n_last_frames=drop_n_last,
        shuffle=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_sampler=batch_sampler,
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare with accelerator
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics,
        initial_step=step, accelerator=accelerator,
    )

    logging.info("Start ER training on merged dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        batch = preprocessor(batch)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.env and eval_env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), accelerator.autocast():
                eval_info = eval_policy_all(
                    envs=eval_env,
                    policy=accelerator.unwrap_model(policy),
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=100,
                    start_seed=cfg.seed,
                )
            aggregated = eval_info["overall"]

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics,
                initial_step=step, accelerator=accelerator,
            )
            eval_tracker.eval_s = aggregated.pop("eval_s")
            eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
            eval_tracker.pc_success = aggregated.pop("pc_success")

            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                if eval_info.get("overall", {}).get("video_paths"):
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][-1], step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir, step, cfg, accelerator.unwrap_model(policy), optimizer, lr_scheduler,
                preprocessor=preprocessor, postprocessor=postprocessor,
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

    if eval_env:
        close_envs(eval_env)

    logging.info("End of training")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    train()
