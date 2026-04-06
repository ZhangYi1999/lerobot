#!/usr/bin/env python

"""Training script with gradient accumulation support.

Drop-in replacement for lerobot_train.py that properly handles gradient accumulation
via HuggingFace Accelerate. Configure accumulation steps through accelerate:

    # Option A: accelerate config → set gradient_accumulation_steps in YAML
    accelerate launch -m lerobot.scripts.lerobot_train_gradient_accumulation ...

    # Option B: env var (recommended — works reliably across accelerate versions)
    GRADIENT_ACCUMULATION_STEPS=8 accelerate launch \
        -m lerobot.scripts.lerobot_train_gradient_accumulation ...

Semantics:
    - cfg.steps = number of optimizer steps (actual weight updates)
    - Total micro-batches = cfg.steps × gradient_accumulation_steps
    - Effective batch size = batch_size × num_processes × gradient_accumulation_steps
    - log_freq, save_freq, eval_freq all count optimizer steps
    - Default gradient_accumulation_steps = 2 (override via GRADIENT_ACCUMULATION_STEPS env var)
"""

import copy
import dataclasses
import logging
import os
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
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
    inside_slurm,
)


# When set to true, merge LoRA adapter weights into the base model before saving.
# This produces a standard (non-PEFT) checkpoint that can be loaded via --policy.pretrained_path.
# Useful for SeqLoRA where each task's merged model becomes the next task's pretrained base.
# Example: export MERGE_LORA_ADAPTER=true
MERGE_LORA_ADAPTER: bool = os.environ.get("MERGE_LORA_ADAPTER", "false").lower() == "true"


def _is_peft_model(model) -> bool:
    """Check if a model is wrapped with PEFT (regardless of how it was configured)."""
    try:
        from peft import PeftModel
        return isinstance(model, PeftModel)
    except ImportError:
        return False


def update_policy_accum(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """Single training micro-step, accumulation-aware.

    Unlike the original update_policy, this function:
    - Only clips gradients on sync steps (after all micro-batches accumulated)
    - Only steps lr_scheduler on sync steps
    - Only calls policy.update() on sync steps
    optimizer.step() and optimizer.zero_grad() are wrapped by accelerator.accumulate()
    and automatically become no-ops on non-sync steps.
    """
    start_time = time.perf_counter()
    policy.train()

    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    with accelerator.autocast():
        if rabc_batch_weights is not None:
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)

    # Only clip gradients on sync step (after all micro-batches accumulated)
    grad_norm = None
    if accelerator.sync_gradients:
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=False
            )

    # Wrapped by accumulate(): no-op on non-sync steps
    with lock if lock is not None else nullcontext():
        optimizer.step()
    optimizer.zero_grad()

    # Only step scheduler and update policy buffers on actual optimizer steps
    if accelerator.sync_gradients:
        if lr_scheduler is not None:
            lr_scheduler.step()
        if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    if grad_norm is not None:
        train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    cfg.validate()

    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        force_cpu = cfg.policy.device == "cpu"
        grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 2))
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
            gradient_accumulation_steps=grad_accum,
        )

    grad_accum_steps = accelerator.gradient_accumulation_steps

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()
    if not is_main_process:
        dataset = make_dataset(cfg)

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    accelerator.wait_for_everyone()

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
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

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")
        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # optimizer steps completed
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes * grad_accum_steps
        total_micro_steps = cfg.steps * grad_accum_steps
        logging.info(
            colored("Gradient accumulation:", "cyan", attrs=["bold"])
            + f" {grad_accum_steps} micro-batches per optimizer step"
        )
        logging.info(
            f"Micro batch size: {cfg.batch_size}, "
            f"Effective batch size: {cfg.batch_size} x {num_processes} x {grad_accum_steps} = {effective_bs}"
        )
        logging.info(
            f"Optimizer steps: {cfg.steps} ({format_big_number(cfg.steps)}), "
            f"Total micro-batches: {total_micro_steps} ({format_big_number(total_micro_steps)})"
        )
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

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

    accelerator.wait_for_everyone()
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
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    optimizer_step = step  # tracks actual weight updates

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - optimizer_step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        effective_batch_size = cfg.batch_size * accelerator.num_processes * grad_accum_steps
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    total_micro_steps = cfg.steps * grad_accum_steps
    for _micro in range(optimizer_step * grad_accum_steps, total_micro_steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        with accelerator.accumulate(policy):
            train_tracker, output_dict = update_policy_accum(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler,
                rabc_weights_provider=rabc_weights,
            )

        # Only act on actual optimizer steps
        if not accelerator.sync_gradients:
            continue

        optimizer_step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and optimizer_step % cfg.log_freq == 0 and is_main_process
        is_saving_step = optimizer_step % cfg.save_freq == 0 or optimizer_step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and optimizer_step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, optimizer_step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {optimizer_step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, optimizer_step)

                unwrapped_policy = accelerator.unwrap_model(policy)
                if MERGE_LORA_ADAPTER and _is_peft_model(unwrapped_policy):
                    logging.info("MERGE_LORA_ADAPTER: merging LoRA adapter into base model")
                    # Save adapter weights separately for reference
                    policy_copy = copy.deepcopy(unwrapped_policy)
                    policy_copy.save_pretrained(str(checkpoint_dir / "adapter"))
                    # Merge LoRA weights into base model
                    merged_model = policy_copy.merge_and_unload()
                    merged_model.config.use_peft = False
                    # Save as a standard (non-PEFT) checkpoint
                    cfg_copy = copy.deepcopy(cfg)
                    cfg_copy.peft = None
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=optimizer_step,
                        cfg=cfg_copy,
                        policy=merged_model,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                    # Explicitly re-save config to ensure 'type' discriminator is present
                    from lerobot.utils.constants import PRETRAINED_MODEL_DIR
                    merged_model.config.save_pretrained(checkpoint_dir / PRETRAINED_MODEL_DIR)
                    del policy_copy, merged_model, cfg_copy
                else:
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=optimizer_step,
                        cfg=cfg,
                        policy=unwrapped_policy,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(optimizer_step, cfg.steps)
                logging.info(f"Eval policy at step {optimizer_step}")
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
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                aggregated = eval_info["overall"]
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=optimizer_step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, optimizer_step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], optimizer_step, mode="eval")
            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")
        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if MERGE_LORA_ADAPTER and _is_peft_model(unwrapped_policy):
                logging.info("MERGE_LORA_ADAPTER: merging LoRA adapter before pushing to hub")
                policy_copy = copy.deepcopy(unwrapped_policy)
                merged_model = policy_copy.merge_and_unload()
                merged_model.config.use_peft = False
                merged_model.push_model_to_hub(cfg)
                del policy_copy, merged_model
            elif _is_peft_model(unwrapped_policy):
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
