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
"""EWC (Elastic Weight Consolidation) baseline for continual learning.

Each task is trained in a separate script invocation. EWC state (Fisher matrix +
parameter checkpoint) is persisted to disk between tasks via ewc_state_path /
ewc_save_path.

Usage:
    # Task 1 (no prior state):
    python ewc.py --config <cfg> --ewc_save_path outputs/ewc_state_task1.pt

    # Task 2 (with prior state):
    python ewc.py --config <cfg> \\
        --policy.pretrained_path outputs/task1/last_checkpoint \\
        --ewc_state_path outputs/ewc_state_task1.pt \\
        --ewc_save_path outputs/ewc_state_task2.pt
"""
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim.optimizers import OptimizerConfig, AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig, DiffuserSchedulerConfig
from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps, IMAGENET_STATS
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransforms
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
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


@dataclass
class EWCTrainPipelineConfig(TrainPipelineConfig):
    # GR00T training defaults
    seed: int | None = 42
    batch_size: int = 32
    steps: int = 10_000
    log_freq: int = 100
    save_freq: int = 10_000
    eval_freq: int = 10_000
    use_policy_training_preset: bool = False
    optimizer: OptimizerConfig | None = field(
        default_factory=lambda: AdamWConfig(
            lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-5, eps=1e-8, grad_clip_norm=1.0
        )
    )
    scheduler: LRSchedulerConfig | None = field(
        default_factory=lambda: DiffuserSchedulerConfig(name="cosine", num_warmup_steps=500)
    )
    # EWC-specific
    ewc_lambda: float = 50000.0       # penalty weight (same as LIBERO/GR00T default)
    ewc_gamma: float = 0.9            # Fisher decay for online EWC across tasks
    ewc_state_path: str | None = None  # load previous Fisher + checkpoint (.pt)
    ewc_fisher_batches: int = 200     # batches used to estimate Fisher after training
    ewc_save_path: str | None = None  # where to save updated EWC state after training

    max_episodes_rendered: int = 100


# ---------------------------------------------------------------------------
# EWC helper utilities
# ---------------------------------------------------------------------------

def get_trainable_params(policy: PreTrainedPolicy) -> list[torch.Tensor]:
    """Return list of parameter tensors that require gradients."""
    return [p for p in policy.parameters() if p.requires_grad]


def flatten_params(params: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of tensors into a single flat 1-D vector."""
    return torch.cat([p.reshape(-1) for p in params])


def compute_ewc_penalty(
    policy: PreTrainedPolicy,
    fish: torch.Tensor,
    checkpoint: torch.Tensor,
) -> torch.Tensor:
    """Diagonal EWC penalty: sum(fish * (theta - theta*)^2)."""
    current = flatten_params(get_trainable_params(policy))
    return (fish * (current - checkpoint).pow(2)).sum()


def load_ewc_state(path: str, device: torch.device) -> dict:
    """Load EWC state dict from disk and move tensors to device."""
    state = torch.load(path, map_location=device, weights_only=True)
    return state


def save_ewc_state(
    fish: torch.Tensor,
    checkpoint: torch.Tensor,
    task_count: int,
    path: str,
) -> None:
    """Save EWC state dict to disk."""
    torch.save(
        {"fish": fish.cpu(), "checkpoint": checkpoint.cpu(), "task_count": task_count},
        path,
    )
    logging.info(f"EWC state saved to {path} (task_count={task_count})")


def compute_fisher(
    policy: PreTrainedPolicy,
    dataloader: torch.utils.data.DataLoader,
    preprocessor,
    accelerator: Accelerator,
    n_batches: int,
) -> torch.Tensor:
    """Estimate diagonal Fisher information via squared gradients.

    Runs a forward+backward pass over up to n_batches from dataloader,
    accumulating grad^2 for each trainable parameter.
    """
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    unwrapped.train()

    trainable_params = get_trainable_params(unwrapped)
    fish = torch.zeros_like(flatten_params(trainable_params))

    actual_batches = 0
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        batch = preprocessor(batch)

        # Zero grads on the unwrapped model directly
        for p in trainable_params:
            if p.grad is not None:
                p.grad.zero_()

        with accelerator.autocast():
            loss, _ = unwrapped.forward(batch)
        accelerator.backward(loss)

        grads = [
            p.grad.detach() if p.grad is not None else torch.zeros_like(p)
            for p in trainable_params
        ]
        fish += flatten_params(grads).pow(2)
        actual_batches += 1

    if actual_batches > 0:
        fish /= actual_batches

    # Clean up gradients to avoid interfering with subsequent optimizer steps
    for p in trainable_params:
        if p.grad is not None:
            p.grad.zero_()

    logging.info(f"Fisher estimated over {actual_batches} batches.")
    return fish


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    fish: torch.Tensor | None,
    checkpoint: torch.Tensor | None,
    ewc_lambda: float,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()

    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    ewc_penalty_value = 0.0
    if fish is not None and checkpoint is not None:
        unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
        ewc_penalty = compute_ewc_penalty(unwrapped, fish, checkpoint)
        loss = loss + ewc_lambda * ewc_penalty
        ewc_penalty_value = ewc_penalty.item()

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

    if output_dict is None:
        output_dict = {}
    output_dict["ewc_penalty"] = ewc_penalty_value

    return train_metrics, output_dict


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

@parser.wrap()
def train(cfg: EWCTrainPipelineConfig):
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

    # ------------------------------------------------------------------
    # Load previous EWC state (if available)
    # ------------------------------------------------------------------
    fish = None
    ewc_checkpoint = None
    task_count = 0

    if cfg.ewc_state_path is not None:
        logging.info(f"Loading EWC state from {cfg.ewc_state_path}")
        ewc_state = load_ewc_state(cfg.ewc_state_path, device)
        fish = ewc_state["fish"].to(device)
        ewc_checkpoint = ewc_state["checkpoint"].to(device)
        task_count = ewc_state["task_count"]
        logging.info(f"Loaded EWC state: task_count={task_count}, fish.shape={fish.shape}")
    else:
        logging.info("No previous EWC state — training task 1 from scratch.")

    # ------------------------------------------------------------------
    # Dataset, policy, optimizer
    # ------------------------------------------------------------------
    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

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
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
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
    logging.info(f"EWC lambda={cfg.ewc_lambda}, gamma={cfg.ewc_gamma}, fisher_batches={cfg.ewc_fisher_batches}")

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
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

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logging.info("Start offline training on a fixed dataset")
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
            fish=fish,
            checkpoint=ewc_checkpoint,
            ewc_lambda=cfg.ewc_lambda,
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
                    max_episodes_rendered=cfg.max_episodes_rendered,
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

    # ------------------------------------------------------------------
    # Post-training: compute Fisher + accumulate + save EWC state
    # ------------------------------------------------------------------
    if cfg.ewc_save_path is not None:
        logging.info("Computing Fisher information matrix on current task data...")

        # Build a fresh (non-cycled) dataloader for Fisher estimation
        fisher_dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=True,
            sampler=None,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )
        fisher_dataloader = accelerator.prepare(fisher_dataloader)

        new_fish = compute_fisher(
            policy,
            fisher_dataloader,
            preprocessor,
            accelerator,
            cfg.ewc_fisher_batches,
        )

        # Online EWC: accumulate with gamma decay
        if fish is not None:
            accumulated_fish = cfg.ewc_gamma * fish + new_fish
            logging.info(f"Accumulated Fisher with gamma={cfg.ewc_gamma}")
        else:
            accumulated_fish = new_fish

        # Capture parameter checkpoint
        unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
        new_checkpoint = flatten_params(get_trainable_params(unwrapped)).detach()

        save_ewc_state(
            accumulated_fish,
            new_checkpoint,
            task_count + 1,
            cfg.ewc_save_path,
        )
    else:
        logging.info(
            "ewc_save_path not set — skipping Fisher computation and EWC state save."
        )

    if eval_env:
        close_envs(eval_env)

    logging.info("End of training")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    train()
