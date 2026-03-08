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
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps, IMAGENET_STATS
from lerobot.optim.optimizers import OptimizerConfig, AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig, DiffuserSchedulerConfig
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
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
class ERTrainPipelineConfig(TrainPipelineConfig):
    # GR00T training defaults
    seed: int | None = 42
    batch_size: int = 16                    # 16 current + 16 replay = 32 total
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
    # ER-specific
    replay_dataset: DatasetConfig | None = None
    replay_num_workers: int = 16
    replay_batch_size: int = 16             # matches batch_size above

    max_episodes_rendered: int = 100



def make_replay_dataset(cfg: ERTrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (ERTrainPipelineConfig): Config which contains a replay DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.replay_dataset.image_transforms) if cfg.replay_dataset.image_transforms.enable else None
    )

    if isinstance(cfg.replay_dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.replay_dataset.repo_id, root=cfg.replay_dataset.root, revision=cfg.replay_dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.replay_dataset.repo_id,
            root=cfg.replay_dataset.root,
            episodes=cfg.replay_dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.replay_dataset.revision,
            video_backend=cfg.replay_dataset.video_backend,
        )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")

    if cfg.replay_dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


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
def train(cfg: ERTrainPipelineConfig):
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

    logging.info("Creating replay buffer dataset")
    replay_dataset = make_replay_dataset(cfg)

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

    # Create dataloaders
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        replay_sampler = EpisodeAwareSampler(
            replay_dataset.meta.episodes["dataset_from_index"],
            replay_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=replay_dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
        replay_sampler = None

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

    replay_dataloader = torch.utils.data.DataLoader(
        replay_dataset,
        num_workers=cfg.replay_num_workers,
        batch_size=cfg.replay_batch_size,
        shuffle=shuffle,
        sampler=replay_sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.replay_num_workers > 0 else None,
    )

    # Prepare with accelerator
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    replay_dataloader = accelerator.prepare(replay_dataloader)
    dl_iter = cycle(dataloader)
    replay_dl_iter = cycle(replay_dataloader)

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

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        replay_batch = next(replay_dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        batch = preprocessor(batch)
        replay_batch = preprocessor(replay_batch)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = torch.cat([batch[key], replay_batch[key]], dim=0)
            else:
                batch[key].extend(replay_batch[key])

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

    if eval_env:
        close_envs(eval_env)

    logging.info("End of training")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    train()
