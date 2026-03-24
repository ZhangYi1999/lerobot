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
import gc
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRScheduler, DiffuserSchedulerConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed, load_rng_state, save_rng_state
from lerobot.utils.train_utils import (
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    save_training_step,
    load_training_step,
)
from lerobot.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.optim.schedulers import load_scheduler_state, save_scheduler_state
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)

from peft import get_peft_model, PeftConfig, PeftModel
from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING

# Controls whether to reuse normalization stats from pretrained checkpoint.
# Default True: when pretrained_path is given, preserve checkpoint's normalizer stats.
# Override: set env var REUSE_PRETRAINED_NORMALIZATION=false to use dataset stats instead.
REUSE_PRETRAINED_NORMALIZATION: bool = os.environ.get("REUSE_PRETRAINED_NORMALIZATION", "true").lower() != "false"

# When set, load normalizer stats from this JSON file (same format as meta/stats.json).
# Takes priority over REUSE_PRETRAINED_NORMALIZATION.
# Example: export NORM_STATS_FILE="union_stats.json"
NORM_STATS_FILE: str | None = os.environ.get("NORM_STATS_FILE")

# CLARE-specific parameters (passed via environment variables)
CLARE_PHASE: str = os.environ.get("CLARE_PHASE", "full")
PEFT_CFG_PATH: str | None = os.environ.get("PEFT_CFG_PATH")
PEFT_WEIGHT_PATH: str | None = os.environ.get("PEFT_WEIGHT_PATH")
ADAPTER_CHECKPOINT_PATH: str | None = os.environ.get("ADAPTER_CHECKPOINT_PATH")
DISCRIMINATOR_TASK_ID: int = int(os.environ.get("DISCRIMINATOR_TASK_ID", "-1"))

TRAIN_DISCRIMINATORS_STEPS: int = int(os.environ.get("TRAIN_DISCRIMINATORS_STEPS", "2000"))
TRAIN_DISCRIMINATORS_LOG_FREQ: int = int(os.environ.get("TRAIN_DISCRIMINATORS_LOG_FREQ", "50"))
TRAIN_DISCRIMINATORS_SAVE_FREQ: int = int(os.environ.get("TRAIN_DISCRIMINATORS_SAVE_FREQ", "2000"))
TRAIN_DISCRIMINATORS_EVAL_FREQ: int = int(os.environ.get("TRAIN_DISCRIMINATORS_EVAL_FREQ", "2000"))
DISC_CHECKPOINT_PATH: str | None = os.environ.get("DISC_CHECKPOINT_PATH")

EXPAND_THRESHOLD: float = float(os.environ.get("EXPAND_THRESHOLD", "0.0"))
AT_LEAST_EXPAND: str = os.environ.get("AT_LEAST_EXPAND", "shallowest")
MAXIMUM_EXPAND: int = int(os.environ.get("MAXIMUM_EXPAND", "10000"))

DETECT_DISTRIBUTION_SHIFT_STEPS: int = int(os.environ.get("DETECT_DISTRIBUTION_SHIFT_STEPS", "200"))
DETECT_DISTRIBUTION_SHIFT_BATCH_SIZE: int = int(os.environ.get("DETECT_DISTRIBUTION_SHIFT_BATCH_SIZE", "32"))
DETECT_DISTRIBUTION_SHIFT_NUM_WORKERS: int = int(os.environ.get("DETECT_DISTRIBUTION_SHIFT_NUM_WORKERS", "16"))
DETECT_DISTRIBUTION_SHIFT_LOG_FREQ: int = int(os.environ.get("DETECT_DISTRIBUTION_SHIFT_LOG_FREQ", "10"))

MAX_EPISODES_RENDERED: int = int(os.environ.get("MAX_EPISODES_RENDERED", "4"))

TRAIN_DISCRIMINATOR_LR: float = float(os.environ.get("TRAIN_DISCRIMINATOR_LR", "0.0005"))
TRAIN_DISCRIMINATOR_WD: float = float(os.environ.get("TRAIN_DISCRIMINATOR_WD", "0.01"))
TRAIN_DISCRIMINATOR_GRAD_CLIP: float = float(os.environ.get("TRAIN_DISCRIMINATOR_GRAD_CLIP", "10.0"))


def _log_vram(tag: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"[VRAM] {tag}: allocated={alloc:.2f}GB  reserved={reserved:.2f}GB")


ADAPTER_CHECKPOINTS_DIR = "adapter_checkpoints"
DISC_CHECKPOINTS_DIR = "disc_checkpoints"


def get_adapter_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns adapter_checkpoints/XXXXXX/"""
    return Path(output_dir) / ADAPTER_CHECKPOINTS_DIR / get_step_identifier(step, total_steps)


def get_disc_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns disc_checkpoints/XXXXXX/"""
    return Path(output_dir) / DISC_CHECKPOINTS_DIR / get_step_identifier(step, total_steps)


def update_last_adapter_checkpoint(adapter_ckpt_dir: Path, output_dir: Path, local_step: int) -> None:
    """Update adapter_checkpoints/last, checkpoints/{step} symlink, and checkpoints/last."""
    output_dir = Path(output_dir)
    adapter_ckpts = output_dir / ADAPTER_CHECKPOINTS_DIR
    # adapter_checkpoints/last -> step_id
    last_adapter = adapter_ckpts / "last"
    if last_adapter.is_symlink():
        last_adapter.unlink()
    last_adapter.symlink_to(adapter_ckpt_dir.name)
    # checkpoints/{step_id} -> ../adapter_checkpoints/{step_id}
    ckpts = output_dir / "checkpoints"
    ckpts.mkdir(parents=True, exist_ok=True)
    step_link = ckpts / f"{local_step:06d}"
    if step_link.is_symlink():
        step_link.unlink()
    step_link.symlink_to(Path("..") / ADAPTER_CHECKPOINTS_DIR / adapter_ckpt_dir.name)
    # checkpoints/last -> ../adapter_checkpoints/last
    last = ckpts / "last"
    if last.is_symlink():
        last.unlink()
    last.symlink_to(Path("..") / ADAPTER_CHECKPOINTS_DIR / "last")


def update_last_disc_checkpoint(
    disc_ckpt_dir: Path, output_dir: Path, cfg_steps: int, disc_step: int
) -> None:
    """Update disc_checkpoints/last, checkpoints/{global_step} symlink, and checkpoints/last."""
    output_dir = Path(output_dir)
    disc_ckpts = output_dir / DISC_CHECKPOINTS_DIR
    # disc_checkpoints/last -> step_id
    last_disc = disc_ckpts / "last"
    if last_disc.is_symlink():
        last_disc.unlink()
    last_disc.symlink_to(disc_ckpt_dir.name)
    # checkpoints/{global_step_id} -> ../disc_checkpoints/{disc_step_id}
    ckpts = output_dir / "checkpoints"
    ckpts.mkdir(parents=True, exist_ok=True)
    global_step = cfg_steps + disc_step
    step_link = ckpts / f"{global_step:06d}"
    if step_link.is_symlink():
        step_link.unlink()
    step_link.symlink_to(Path("..") / DISC_CHECKPOINTS_DIR / disc_ckpt_dir.name)
    # checkpoints/last -> ../disc_checkpoints/last
    last = ckpts / "last"
    if last.is_symlink():
        last.unlink()
    last.symlink_to(Path("..") / DISC_CHECKPOINTS_DIR / "last")


def save_discriminator_training_state(
    checkpoint_dir: Path, step: int, optimizer: Optimizer, scheduler
) -> None:
    """Save discriminator training state to discriminator_training_state/ subdir.

    Mirrors load_discriminator_training_state() which reads from this same location.
    """
    save_dir = checkpoint_dir / "discriminator_training_state"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(step, save_dir)
    save_rng_state(save_dir)
    save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def _make_discriminator_optimizer_config() -> AdamWConfig:
    return AdamWConfig(
        lr=TRAIN_DISCRIMINATOR_LR,
        weight_decay=TRAIN_DISCRIMINATOR_WD,
        grad_clip_norm=TRAIN_DISCRIMINATOR_GRAD_CLIP,
        betas=(0.9, 0.999),
        eps=1e-08,
    )


def _prefix_keys(d: dict, prefix: str) -> dict:
    """Prefix all keys in dict for WandB logging (mode-safe)."""
    return {f"{prefix}/{k}": v for k, v in d.items() if isinstance(v, (int, float, str))}


def set_peft_module_train(peft_modules: list, train: bool = True):
    prefix = PEFT_TYPE_TO_PREFIX_MAPPING[peft_modules[0].peft_config.peft_type]
    for peft_module in peft_modules:
        for name, module in peft_module.named_modules():
            if prefix in name or name == '':
                module.train(train)
            if 'base_layer' in name:
                module.train(False)
    return peft_modules


def detect_distribution_shift(
    cfg: TrainPipelineConfig,
    wandb_logger: WandBLogger,
    global_steps: int,
    policy: PreTrainedPolicy,
    peft_modules: list,
    dataset,
    preprocessor,
    accelerator: Accelerator,
):
    device = accelerator.device

    infer_metrics = {
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    for peft_module in peft_modules:
        peft_module.track_z_score(True)
        for discriminator_id in range(peft_module.num_discriminators):
            key = f"{peft_module.layer_name}.{peft_module.layer_id}.{discriminator_id}"

            infer_metrics[f"loss_{key}"] = AverageMeter(f"loss_{key}", ":.3f")
            infer_metrics[f"z_score_{key}"] = AverageMeter(f"z_score_{key}", ":.3f")

    detect_tracker = MetricsTracker(
        DETECT_DISTRIBUTION_SHIFT_BATCH_SIZE, dataset.num_frames, dataset.num_episodes,
        infer_metrics, initial_step=0, accelerator=accelerator,
    )

    detect_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=DETECT_DISTRIBUTION_SHIFT_NUM_WORKERS,
        batch_size=DETECT_DISTRIBUTION_SHIFT_BATCH_SIZE,
        shuffle=True,
        sampler=None,
        pin_memory=device.type != "cpu",
        drop_last=False,
        prefetch_factor=2 if DETECT_DISTRIBUTION_SHIFT_NUM_WORKERS > 0 else None,
    )
    detect_iter = cycle(detect_dataloader)

    policy.eval()

    z_scores_sum = {}
    losses_sum = {}

    step = 0

    for _ in range(DETECT_DISTRIBUTION_SHIFT_STEPS):
        batch = next(detect_iter)
        batch = preprocessor(batch)

        with torch.no_grad(), accelerator.autocast():
            _, _ = policy.forward(batch)

        for peft_module in peft_modules:
            info_dicts = peft_module.info_dicts
            for discriminator_id in range(peft_module.num_discriminators):

                discriminator_info_dict = info_dicts[f"discriminator_{discriminator_id}"]

                loss = discriminator_info_dict["loss"].mean().item()
                z_score = discriminator_info_dict["z_score"].mean().item()

                key = f"{peft_module.layer_name}.{peft_module.layer_id}"

                if key in z_scores_sum:
                    if len(z_scores_sum[key]) <= discriminator_id:
                        z_scores_sum[key].append(z_score)
                        losses_sum[key].append(loss)
                    else:
                        z_scores_sum[key][discriminator_id] += z_score
                        losses_sum[key][discriminator_id] += loss
                else:
                    z_scores_sum[key] = [z_score]
                    losses_sum[key] = [loss]

                log_key = f"{key}.{discriminator_id}"
                detect_tracker.__setattr__(f"loss_{log_key}", loss)
                detect_tracker.__setattr__(f"z_score_{log_key}", z_score)

        step += 1
        detect_tracker.step()
        is_log_step = DETECT_DISTRIBUTION_SHIFT_LOG_FREQ > 0 and step % DETECT_DISTRIBUTION_SHIFT_LOG_FREQ == 0

        if is_log_step:
            logging.info(detect_tracker)
            if wandb_logger:
                wandb_log_dict = _prefix_keys(detect_tracker.to_dict(), "detect_shift")
                wandb_step = step
                if global_steps > 0:
                    wandb_step += global_steps
                wandb_logger.log_dict(wandb_log_dict, wandb_step, mode='train')
            detect_tracker.reset_averages()

    z_scores_mean = {}
    losses_mean = {}

    for peft_module in peft_modules:
        peft_module.track_z_score(False)

        key = f"{peft_module.layer_name}.{peft_module.layer_id}"

        z_scores_mean_current_layer = torch.tensor(z_scores_sum[key], device="cpu") / step
        losses_mean_current_layer = torch.tensor(losses_sum[key], device="cpu") / step

        logging.info(f"Distribution shift of {key}")
        logging.info(f"Average z_scores: {[f'{z_score:.4f}' for z_score in z_scores_mean_current_layer.tolist()]}")
        logging.info(f"Average losses: {[f'{loss:.4f}' for loss in losses_mean_current_layer.tolist()]}")

        z_scores_mean[key] = z_scores_mean_current_layer
        losses_mean[key] = losses_mean_current_layer

    return z_scores_mean, losses_mean, global_steps + step


def load_discriminator_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    training_state_dir = checkpoint_dir / "discriminator_training_state"
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    peft_modules: list,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    peft_modules = set_peft_module_train(peft_modules)
    with accelerator.autocast():
        policy_loss, output_dict = policy.forward(batch)

        if peft_modules[0]._train_discriminator:
            discriminators_loss = []
            for peft_module in peft_modules:
                for discriminator_id in range(peft_module.num_discriminators):
                    if f"discriminator_{discriminator_id}" not in peft_module.info_dicts:
                        logging.error(f"Missing discriminator_{discriminator_id} in info_dicts for {peft_module.layer_name}.{peft_module.layer_id}")
                        logging.error(f"  info_dicts keys: {list(peft_module.info_dicts.keys())}")
                        logging.error(f"  _train_discriminator={peft_module._train_discriminator} training={peft_module.training}")
                        logging.error(f"  _forwarded_discriminator_id={peft_module._forwarded_discriminator_id}")
                        logging.error(f"  num_discriminators={peft_module.num_discriminators}")
                        logging.error(f"  clare_discriminators keys={list(peft_module.clare_discriminators.keys())}")
                        if peft_module.adapter_name in peft_module.clare_discriminators:
                            logging.error(f"  num disc modules={len(peft_module.clare_discriminators[peft_module.adapter_name])}")
                    discriminator_info_dict = peft_module.info_dicts[f"discriminator_{discriminator_id}"]

                    discriminator_running_mean = discriminator_info_dict["running_mean"]
                    discriminator_running_std = discriminator_info_dict["running_std"]
                    discriminator_num_batches_tracked = discriminator_info_dict["num_batches_tracked"]

                    key = f"{peft_module.layer_name}.{peft_module.layer_id}.{discriminator_id}"

                    train_metrics.__setattr__(f"running_mean_{key}", discriminator_running_mean.item())
                    train_metrics.__setattr__(f"running_std_{key}", discriminator_running_std.item())
                    train_metrics.__setattr__(f"num_batches_tracked_{key}", discriminator_num_batches_tracked.item())

                    if discriminator_id == peft_module._forwarded_discriminator_id:
                        discriminator_loss = discriminator_info_dict["loss"].mean()
                        discriminators_loss.append(discriminator_loss)
                        train_metrics.__setattr__(f"loss_{key}", discriminator_loss.item())
            loss = sum(discriminators_loss)
        else:
            loss = policy_loss

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


def _setup_common(cfg: TrainPipelineConfig):
    """Shared setup for both adapter and discriminator training phases.

    Returns:
        tuple of (accelerator, device, dataset, eval_env, env_preprocessor, env_postprocessor,
                  policy, peft_policy, peft_modules, peft_config, preprocessor, postprocessor, wandb_logger)
    """
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
    policy.eval()

    # Create preprocessor/postprocessor
    # Resolve normalization stats: NORM_STATS_FILE > dataset stats / pretrained stats
    _norm_stats = None
    if NORM_STATS_FILE:
        import json
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

    logging.info("Wrapping policy with peft module")

    # Determine which PEFT weights to load
    peft_weight_path = PEFT_WEIGHT_PATH
    if CLARE_PHASE == "discriminator" and ADAPTER_CHECKPOINT_PATH:
        peft_weight_path = ADAPTER_CHECKPOINT_PATH

    if peft_weight_path:
        peft_policy = PeftModel.from_pretrained(policy, peft_weight_path, is_trainable=True, autocast_adapter_dtype=False)
        peft_config = peft_policy.peft_config["default"]
    else:
        peft_cfg = PeftConfig.from_pretrained(PEFT_CFG_PATH)
        peft_cfg.inference_mode = False
        peft_policy = get_peft_model(policy, peft_cfg)
        peft_config = peft_policy.peft_config["default"]

    peft_modules = peft_policy.base_model.adapter_layers

    return (accelerator, device, dataset, eval_env, env_preprocessor, env_postprocessor,
            policy, peft_policy, peft_modules, peft_config, preprocessor, postprocessor, wandb_logger)


def _expand_layers(
    cfg: TrainPipelineConfig,
    wandb_logger,
    policy,
    peft_modules,
    peft_config,
    dataset,
    preprocessor,
    accelerator,
):
    """Detect distribution shift and expand layers. Returns (adapter_params, discriminator_params, step)."""
    adapter_params = []
    discriminator_params = []

    new_task_id = peft_config.num_learned_task
    step = 0

    if new_task_id == 0:
        logging.info("Learning the first new task")
        logging.info("Expand all finetuned layers with new adapter and new discriminator")

        for peft_module in peft_modules:
            adapter_param, discriminator_param = \
                peft_module.add_adapter_and_discriminator(new_task_id)
            adapter_params += adapter_param
            discriminator_params += discriminator_param

            key = f"{peft_module.layer_name}.{peft_module.layer_id}"

            peft_module._forwarded_adapter_id = peft_module.num_adapters - 1
            peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1

            peft_config.structure[key] = \
                [peft_module.num_adapters, peft_module.num_discriminators]
            logging.info(f"Add both adapter and discriminator into layer {key}")
            logging.info(f"Only forward adapter id: {peft_module._forwarded_adapter_id}")
            logging.info(f"Only forward discriminator id: {peft_module._forwarded_discriminator_id}")
    else:
        z_scores_mean, losses_mean, step = \
            detect_distribution_shift(
                cfg, wandb_logger, 0, policy, peft_modules, dataset, preprocessor, accelerator
            )

        only_forward_ids = []
        to_expand_or_not = []

        for peft_module in peft_modules:
            key = f"{peft_module.layer_name}.{peft_module.layer_id}"
            logging.info(f"For layer {key}")

            z_scores_mean_current_layer = z_scores_mean[key]
            losses_mean_current_layer = losses_mean[key]

            closest_discriminator_id = torch.argmin(losses_mean_current_layer).item()
            connected_adapter_id = peft_module.get_adapter_id_by_discriminator_id(closest_discriminator_id)
            only_forward_ids.append(connected_adapter_id)

            expand_the_module = False

            if all(z_scores_mean_current_layer > EXPAND_THRESHOLD):
                logging.info(f"All z-scores in layer {key} exceed threshold {EXPAND_THRESHOLD}")
                logging.info(f"Will try to add new adapter and new discriminator in layer {key}")
                expand_the_module = True
            else:
                logging.info(f"At least one z_score in layer {key} is lower than threshold {EXPAND_THRESHOLD}")
                logging.info(f"Will try to only add new discriminator in layer {key}")

            if expand_the_module:
                if sum(to_expand_or_not) < MAXIMUM_EXPAND:
                    logging.info("The number of new adapter is within limit")
                else:
                    logging.info("The number of new adapter reaches the expansion limit")
                    expand_the_module = False

            to_expand_or_not.append(expand_the_module)

        if sum(to_expand_or_not) == 0:
            logging.info("No layer have expansion signal. But still expand")
            if AT_LEAST_EXPAND == "shallowest":
                logging.info("Still expand the shallowest layer.")
                to_expand_or_not[0] = True
                only_forward_ids[0] = -1
            elif AT_LEAST_EXPAND == "deepest":
                logging.info("Still expand the deepest layer.")
                to_expand_or_not[-1] = True
                only_forward_ids[-1] = -1

        for peft_module_id, (to_expand, only_forward_id) in enumerate(zip(to_expand_or_not, only_forward_ids)):
            peft_module = peft_modules[peft_module_id]

            peft_module.train_discriminator(False)
            key = f"{peft_module.layer_name}.{peft_module.layer_id}"
            logging.info(f"For layer {key}")

            if to_expand:
                adapter_param, discriminator_param = \
                    peft_module.add_adapter_and_discriminator(new_task_id)
                adapter_params += adapter_param
                discriminator_params += discriminator_param

                peft_module._forwarded_adapter_id = peft_module.num_adapters - 1
                peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1
                logging.info(f"Add both adapter and discriminator into layer {key}")
            else:
                discriminator_param = \
                    peft_module.add_discriminator(only_forward_id, new_task_id)
                discriminator_params += discriminator_param

                peft_module._forwarded_adapter_id = only_forward_id
                peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1

                attached_adapter_task_id = peft_module.clare_func_adapters[peft_module.adapter_name][only_forward_id].task_id.item()
                logging.info(f"Only add discriminator into layer {key}, attatch it with adapter of task_id {attached_adapter_task_id}")

            logging.info(f"Only forward adapter id: {peft_module._forwarded_adapter_id}")
            logging.info(f"Only forward discriminator id: {peft_module._forwarded_discriminator_id}")
            peft_module._active_task = new_task_id
            peft_config.structure[key] = \
                [peft_module.num_adapters, peft_module.num_discriminators]

    peft_config.num_learned_task += 1

    return adapter_params, discriminator_params, step


def _do_eval(
    cfg, step, policy, peft_modules, eval_env, env_preprocessor, env_postprocessor,
    preprocessor, postprocessor, dataset, wandb_logger, accelerator,
):
    """Run evaluation, temporarily stopping gradients on PEFT modules."""
    step_id = get_step_identifier(step, cfg.steps)
    logging.info(f"Eval policy at step {step}")

    # Stop gradients during evaluation
    to_train_module_list = []
    for peft_module in peft_modules:
        for name, parameter in peft_module.named_parameters():
            if parameter.requires_grad:
                to_train_module_list.append(name)
                parameter.requires_grad = False

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
            max_episodes_rendered=MAX_EPISODES_RENDERED,
            start_seed=cfg.seed,
        )

    # Restore gradients
    for peft_module in peft_modules:
        for name, parameter in peft_module.named_parameters():
            if name in to_train_module_list:
                parameter.requires_grad = True

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


def train_adapter(cfg: TrainPipelineConfig):
    """Phase 1: Expand layers and train func_adapters only."""
    (accelerator, device, dataset, eval_env, env_preprocessor, env_postprocessor,
     policy, peft_policy, peft_modules, peft_config, preprocessor, postprocessor, wandb_logger) = _setup_common(cfg)

    # Expand layers
    adapter_params, discriminator_params, step = _expand_layers(
        cfg, wandb_logger, policy, peft_modules, peft_config, dataset, preprocessor, accelerator
    )

    # Create adapter optimizer
    adapter_optimizer = cfg.optimizer.build(adapter_params)
    if cfg.scheduler:
        adapter_lr_scheduler = cfg.scheduler.build(adapter_optimizer, cfg.steps)
    else:
        adapter_lr_scheduler = None

    # global_step_offset: steps consumed by the detection phase (0 for task 0).
    # local_step: steps within THIS adapter training phase (0 → cfg.steps).
    global_step_offset = step
    local_step = 0
    if cfg.resume:
        local_step, adapter_optimizer, adapter_lr_scheduler = load_training_state(
            cfg.checkpoint_path, adapter_optimizer, adapter_lr_scheduler
        )
    step = global_step_offset + local_step  # global counter for WandB

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_adapter_params = sum(p.numel() for p in adapter_params)
    num_discriminator_params = sum(p.numel() for p in discriminator_params)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_adapter_params=} ({format_big_number(num_adapter_params)})")
    logging.info(f"{num_discriminator_params=} ({format_big_number(num_discriminator_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Create dataloader
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

    # Set adapter training mode
    for peft_module in peft_modules:
        peft_module.train_discriminator(False)
        peft_module.update_stats(False)

    optimizer = adapter_optimizer
    lr_scheduler = adapter_lr_scheduler

    # Prepare with accelerator
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    peft_modules = set_peft_module_train(peft_modules)

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

    logging.info("Start training func adapters")
    for _ in range(local_step, cfg.steps):  # only remaining steps when resuming
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker, policy, peft_modules, batch, optimizer,
            cfg.optimizer.grad_clip_norm, accelerator=accelerator, lr_scheduler=lr_scheduler,
        )

        step += 1
        local_step += 1
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = local_step % cfg.save_freq == 0 or local_step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and local_step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step, mode="train")
            train_tracker.reset_averages()

        if cfg.env and eval_env and is_eval_step:
            _do_eval(cfg, step, policy, peft_modules, eval_env, env_preprocessor, env_postprocessor,
                     preprocessor, postprocessor, dataset, wandb_logger, accelerator)

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after adapter step {local_step}")
            checkpoint_dir = get_adapter_step_checkpoint_dir(cfg.output_dir, cfg.steps, local_step)

            peft_policy.save_pretrained(str(checkpoint_dir / "adapter"))

            save_checkpoint(
                checkpoint_dir, local_step, cfg, accelerator.unwrap_model(policy), optimizer, lr_scheduler,
                preprocessor=preprocessor, postprocessor=postprocessor,
            )
            update_last_adapter_checkpoint(checkpoint_dir, cfg.output_dir, local_step)

    if eval_env:
        close_envs(eval_env)

    logging.info("End of adapter training")
    _log_vram("train_adapter end")

    accelerator.wait_for_everyone()
    accelerator.end_training()


def train_discriminator(cfg: TrainPipelineConfig, skip_expand: bool = False):
    """Phase 2: Train discriminators only (loads adapter checkpoint)."""
    (accelerator, device, dataset, eval_env, env_preprocessor, env_postprocessor,
     policy, peft_policy, peft_modules, peft_config, preprocessor, postprocessor, wandb_logger) = _setup_common(cfg)

    if skip_expand:
        # In "full" mode, layers are already expanded — just collect existing discriminator params
        discriminator_params = []
        for peft_module in peft_modules:
            for disc in peft_module.clare_discriminators[peft_module.adapter_name]:
                for p in disc.parameters():
                    p.requires_grad = True
                discriminator_params += list(disc.parameters())
            # Set forwarded IDs to the last (current task's) adapter/discriminator
            peft_module._forwarded_adapter_id = peft_module.num_adapters - 1
            peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1
    else:
        adapter_params, discriminator_params, _detection_steps = _expand_layers(
            cfg, wandb_logger, policy, peft_modules, peft_config, dataset, preprocessor, accelerator
        )

    # Create discriminator optimizer
    disc_optimizer_cfg = _make_discriminator_optimizer_config()
    discriminator_optimizer = disc_optimizer_cfg.build(discriminator_params)
    discriminator_lr_scheduler = None

    disc_step = 0
    if cfg.resume:
        assert DISC_CHECKPOINT_PATH, "DISC_CHECKPOINT_PATH must be set when resuming discriminator training"
        disc_step, discriminator_optimizer, discriminator_lr_scheduler = load_discriminator_training_state(
            Path(DISC_CHECKPOINT_PATH), discriminator_optimizer, discriminator_lr_scheduler
        )

    # Set discriminator training mode
    for peft_module in peft_modules:
        peft_module.train_discriminator(True)
        peft_module.update_stats(True)

    # Freeze everything except discriminator params so autograd does not build a
    # computation graph through the adapter during forward passes (saves activation memory).
    for p in policy.parameters():
        p.requires_grad = False
    for p in discriminator_params:
        p.requires_grad = True
    _log_vram("train_discriminator start (after freeze)")

    optimizer = discriminator_optimizer
    lr_scheduler = discriminator_lr_scheduler

    # Create dataloader
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

    peft_modules = set_peft_module_train(peft_modules)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    for peft_module in peft_modules:
        for discriminator_id in range(peft_module.num_discriminators):
            key = f"{peft_module.layer_name}.{peft_module.layer_id}.{discriminator_id}"
            train_metrics[f"loss_{key}"] = AverageMeter(f"loss_{key}", ":.3f")
            train_metrics[f"running_mean_{key}"] = AverageMeter(f"running_mean_{key}", ":.3f")
            train_metrics[f"running_std_{key}"] = AverageMeter(f"running_std_{key}", ":.3f")
            train_metrics[f"num_batches_tracked_{key}"] = AverageMeter(f"num_batches_tracked_{key}", ":.0f")

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics,
        initial_step=disc_step, accelerator=accelerator,
    )

    logging.info("Start training discriminators")
    for _ in range(disc_step, TRAIN_DISCRIMINATORS_STEPS):  # only remaining steps when resuming
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker, policy, peft_modules, batch, optimizer,
            disc_optimizer_cfg.grad_clip_norm, accelerator=accelerator, lr_scheduler=lr_scheduler,
        )

        disc_step += 1
        train_tracker.step()

        if disc_step == 1:
            _log_vram("train_discriminator first step")

        is_log_step = TRAIN_DISCRIMINATORS_LOG_FREQ > 0 and disc_step % TRAIN_DISCRIMINATORS_LOG_FREQ == 0
        is_saving_step = disc_step % TRAIN_DISCRIMINATORS_SAVE_FREQ == 0 or disc_step == TRAIN_DISCRIMINATORS_STEPS
        is_eval_step = TRAIN_DISCRIMINATORS_EVAL_FREQ > 0 and disc_step % TRAIN_DISCRIMINATORS_EVAL_FREQ == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = _prefix_keys(train_tracker.to_dict(), "discriminator")
                if output_dict:
                    wandb_log_dict.update(_prefix_keys(output_dict, "discriminator"))
                wandb_logger.log_dict(wandb_log_dict, disc_step, mode="train")
            train_tracker.reset_averages()

        if cfg.env and eval_env and is_eval_step:
            _do_eval(cfg, disc_step, policy, peft_modules, eval_env, env_preprocessor, env_postprocessor,
                     preprocessor, postprocessor, dataset, wandb_logger, accelerator)

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after discriminator step {disc_step}")
            checkpoint_dir = get_disc_step_checkpoint_dir(cfg.output_dir, TRAIN_DISCRIMINATORS_STEPS, disc_step)

            peft_policy.save_pretrained(str(checkpoint_dir / "adapter"))

            save_checkpoint(
                checkpoint_dir, disc_step, cfg, accelerator.unwrap_model(policy), optimizer, lr_scheduler,
                preprocessor=preprocessor, postprocessor=postprocessor,
            )
            save_discriminator_training_state(checkpoint_dir, disc_step, optimizer, lr_scheduler)
            update_last_disc_checkpoint(checkpoint_dir, cfg.output_dir, cfg.steps, disc_step)

    if eval_env:
        close_envs(eval_env)

    logging.info("End of discriminator training")

    accelerator.wait_for_everyone()
    accelerator.end_training()


def train_discriminator_only(cfg: TrainPipelineConfig):
    """Train discriminators for a single task on a pre-converted CLARE checkpoint.

    Unlike train_discriminator(), this skips _expand_layers() because adapters and
    discriminators already exist in the checkpoint (from convert_lora_to_clare.py).
    Only trains the discriminator for cfg.discriminator_task_id.
    """
    (accelerator, device, dataset, eval_env, env_preprocessor, env_postprocessor,
     policy, peft_policy, peft_modules, peft_config, preprocessor, postprocessor, wandb_logger) = _setup_common(cfg)

    task_id = DISCRIMINATOR_TASK_ID
    logging.info(f"Training discriminator for task {task_id} (discriminator_only mode)")

    # Set up forwarding: each CLARELayer forwards the adapter and discriminator for this task
    discriminator_params = []
    for peft_module in peft_modules:
        if task_id >= peft_module.num_discriminators:
            raise ValueError(
                f"discriminator_task_id={task_id} but CLARELayer "
                f"{peft_module.layer_name}.{peft_module.layer_id} only has "
                f"{peft_module.num_discriminators} discriminators"
            )
        peft_module._forwarded_adapter_id = task_id
        peft_module._forwarded_discriminator_id = task_id
        peft_module._active_task = task_id
        peft_module.train_discriminator(True)
        peft_module.update_stats(True)

        # Collect discriminator parameters for this task
        disc = peft_module.clare_discriminators[peft_module.adapter_name][task_id]
        for p in disc.parameters():
            p.requires_grad = True
        discriminator_params.extend(list(disc.parameters()))

    logging.info(f"Collected {len(discriminator_params)} discriminator parameters")

    # Freeze everything except discriminator params so autograd does not build a
    # computation graph through the adapter during forward passes (saves activation memory).
    for p in policy.parameters():
        p.requires_grad = False
    for p in discriminator_params:
        p.requires_grad = True

    # Create discriminator optimizer
    disc_optimizer_cfg = _make_discriminator_optimizer_config()
    discriminator_optimizer = disc_optimizer_cfg.build(discriminator_params)
    discriminator_lr_scheduler = None

    disc_step = 0
    if cfg.resume:
        assert DISC_CHECKPOINT_PATH, "DISC_CHECKPOINT_PATH must be set when resuming discriminator_only training"
        disc_step, discriminator_optimizer, discriminator_lr_scheduler = load_discriminator_training_state(
            Path(DISC_CHECKPOINT_PATH), discriminator_optimizer, discriminator_lr_scheduler
        )

    optimizer = discriminator_optimizer
    lr_scheduler = discriminator_lr_scheduler

    # Create dataloader
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

    peft_modules = set_peft_module_train(peft_modules)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    for peft_module in peft_modules:
        for discriminator_id in range(peft_module.num_discriminators):
            key = f"{peft_module.layer_name}.{peft_module.layer_id}.{discriminator_id}"
            train_metrics[f"loss_{key}"] = AverageMeter(f"loss_{key}", ":.3f")
            train_metrics[f"running_mean_{key}"] = AverageMeter(f"running_mean_{key}", ":.3f")
            train_metrics[f"running_std_{key}"] = AverageMeter(f"running_std_{key}", ":.3f")
            train_metrics[f"num_batches_tracked_{key}"] = AverageMeter(f"num_batches_tracked_{key}", ":.0f")

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics,
        initial_step=disc_step, accelerator=accelerator,
    )

    logging.info(f"Start training discriminator for task {task_id}")
    for _ in range(disc_step, TRAIN_DISCRIMINATORS_STEPS):  # only remaining steps when resuming
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker, policy, peft_modules, batch, optimizer,
            disc_optimizer_cfg.grad_clip_norm, accelerator=accelerator, lr_scheduler=lr_scheduler,
        )

        disc_step += 1
        train_tracker.step()

        is_log_step = TRAIN_DISCRIMINATORS_LOG_FREQ > 0 and disc_step % TRAIN_DISCRIMINATORS_LOG_FREQ == 0
        is_saving_step = disc_step % TRAIN_DISCRIMINATORS_SAVE_FREQ == 0 or disc_step == TRAIN_DISCRIMINATORS_STEPS
        is_eval_step = TRAIN_DISCRIMINATORS_EVAL_FREQ > 0 and disc_step % TRAIN_DISCRIMINATORS_EVAL_FREQ == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = _prefix_keys(train_tracker.to_dict(), f"discriminator_task{task_id}")
                if output_dict:
                    wandb_log_dict.update(_prefix_keys(output_dict, f"discriminator_task{task_id}"))
                wandb_logger.log_dict(wandb_log_dict, disc_step, mode="train")
            train_tracker.reset_averages()

        if cfg.env and eval_env and is_eval_step:
            _do_eval(cfg, disc_step, policy, peft_modules, eval_env, env_preprocessor, env_postprocessor,
                     preprocessor, postprocessor, dataset, wandb_logger, accelerator)

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after discriminator step {disc_step}")
            checkpoint_dir = get_disc_step_checkpoint_dir(cfg.output_dir, TRAIN_DISCRIMINATORS_STEPS, disc_step)

            peft_policy.save_pretrained(str(checkpoint_dir / "adapter"))

            save_checkpoint(
                checkpoint_dir, disc_step, cfg, accelerator.unwrap_model(policy), optimizer, lr_scheduler,
                preprocessor=preprocessor, postprocessor=postprocessor,
            )
            save_discriminator_training_state(checkpoint_dir, disc_step, optimizer, lr_scheduler)
            update_last_disc_checkpoint(checkpoint_dir, cfg.output_dir, cfg.steps, disc_step)

    if eval_env:
        close_envs(eval_env)

    logging.info(f"End of discriminator training for task {task_id}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    global PEFT_WEIGHT_PATH, PEFT_CFG_PATH

    # CLARE uses fixed optimizer/scheduler defaults when policy preset is not used
    if not cfg.use_policy_training_preset:
        if cfg.optimizer is None:
            cfg.optimizer = AdamWConfig(
                lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-5, eps=1e-8, grad_clip_norm=1.0
            )
        if cfg.scheduler is None:
            cfg.scheduler = DiffuserSchedulerConfig(name="cosine", num_warmup_steps=500)

    # Validate CLARE env vars
    assert CLARE_PHASE in ("full", "adapter", "discriminator", "discriminator_only"), \
        f"CLARE_PHASE must be full/adapter/discriminator/discriminator_only, got {CLARE_PHASE}"
    assert PEFT_CFG_PATH or PEFT_WEIGHT_PATH, \
        "One of PEFT_CFG_PATH or PEFT_WEIGHT_PATH env var must be set"
    if CLARE_PHASE == "discriminator_only":
        assert DISCRIMINATOR_TASK_ID >= 0, \
            "DISCRIMINATOR_TASK_ID must be >= 0 for discriminator_only phase"
        assert PEFT_WEIGHT_PATH, \
            "PEFT_WEIGHT_PATH must be set for discriminator_only phase"

    cfg.validate()

    if CLARE_PHASE == "adapter":
        train_adapter(cfg)
    elif CLARE_PHASE == "discriminator":
        train_discriminator(cfg)
    elif CLARE_PHASE == "discriminator_only":
        train_discriminator_only(cfg)
    else:  # "full" — original behavior
        train_adapter(cfg)
        # Free adapter optimizer states (Adam m/v) before discriminator phase.
        # train_adapter's locals are out of scope but PyTorch's caching allocator
        # holds VRAM until explicitly flushed.
        _log_vram("before gc/empty_cache")
        gc.collect()
        torch.cuda.empty_cache()
        _log_vram("after gc/empty_cache")
        # Point to the adapter checkpoint so train_discriminator loads trained weights
        adapter_ckpt = (Path(cfg.output_dir) / "checkpoints" / "last" / "adapter").resolve()
        PEFT_WEIGHT_PATH = str(adapter_ckpt)
        PEFT_CFG_PATH = None
        train_discriminator(cfg, skip_expand=True)


if __name__ == "__main__":
    train()
