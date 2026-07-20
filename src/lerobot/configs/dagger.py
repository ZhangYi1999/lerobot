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

import datetime as dt
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from lerobot import envs, policies  # noqa: F401

from . import parser
from .policies import PreTrainedConfig

logger = getLogger(__name__)

# Available beta-decay schedules for the mixed rollout policy.
BETA_SCHEDULES = ("exponential", "linear", "constant")


@dataclass
class DAggerConfig:
    """Hyper-parameters for the DAgger (Dataset Aggregation) loop.

    A DAgger run alternates between two phases for ``n_iterations`` rounds:
    1. Collection: the student rolls out in the environment while the expert relabels every visited
       state. Executed actions are drawn from the expert with probability ``beta`` (mixed rollout),
       so the state distribution shifts from expert- to student-induced as ``beta`` decays.
    2. Training: the student is optimized for ``steps_per_iteration`` gradient steps on the growing
       aggregated dataset of expert-labeled frames.
    """

    # Number of DAgger rounds (collect + train).
    n_iterations: int = 20

    # ---- Rollout / data collection ----
    # Number of episodes collected per iteration and appended to the aggregated dataset.
    episodes_per_iteration: int = 10
    # Number of parallel environments used during collection. 0 = reuse `episodes_per_iteration`.
    rollout_batch_size: int = 0
    use_async_envs: bool = True
    # Keep only the frames on which the expert action was actually executed (classic on-policy
    # DAgger keeps all visited states, which is the default).
    only_save_expert: bool = False
    # Keep only frames from successful episodes.
    only_success: bool = False

    # ---- Student training ----
    # Gradient steps run on the aggregated dataset each iteration.
    steps_per_iteration: int = 500
    batch_size: int = 8
    num_workers: int = 4
    grad_clip_norm: float = 10.0
    # Log the running training loss every `log_freq` gradient steps.
    log_freq: int = 50

    # ---- Beta scheduling (expert action probability) ----
    init_beta: float = 1.0
    beta_schedule: str = "exponential"
    beta_decay: float = 0.99
    beta_min: float = 0.05

    # ---- Evaluation (pure student) ----
    # Run a pure-student evaluation every `eval_freq` iterations (0 disables).
    eval_freq: int = 1
    eval_n_episodes: int = 20

    def __post_init__(self) -> None:
        if self.beta_schedule not in BETA_SCHEDULES:
            raise ValueError(
                f"dagger.beta_schedule must be one of {BETA_SCHEDULES}, got '{self.beta_schedule}'."
            )
        if not 0.0 <= self.beta_min <= self.init_beta <= 1.0:
            raise ValueError(
                "dagger requires 0.0 <= beta_min <= init_beta <= 1.0, "
                f"got beta_min={self.beta_min}, init_beta={self.init_beta}."
            )
        if not 0.0 < self.beta_decay <= 1.0:
            raise ValueError(f"dagger.beta_decay must be in (0.0, 1.0], got {self.beta_decay}.")
        if self.rollout_batch_size == 0:
            self.rollout_batch_size = self.episodes_per_iteration
        if self.rollout_batch_size > self.episodes_per_iteration:
            self.rollout_batch_size = self.episodes_per_iteration


@dataclass
class DAggerPipelineConfig:
    """Top-level configuration for `lerobot-dagger`.

    `policy` is the *student* being trained; `expert_policy` is a frozen teacher used to relabel the
    states the student visits. Both are ordinary LeRobot policies loaded from a pretrained path.
    """

    env: envs.EnvConfig
    # Student policy (trained in place). Provide with `--policy.path=...`.
    policy: PreTrainedConfig | None = None
    # Expert policy (frozen relabeler). Provide with `--expert_policy.path=...`.
    expert_policy: PreTrainedConfig | None = None
    dagger: DAggerConfig = field(default_factory=DAggerConfig)

    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000

    # Rename maps to override image/state keys for the student and the expert respectively.
    rename_map: dict[str, str] = field(default_factory=dict)
    expert_rename_map: dict[str, str] = field(default_factory=dict)

    # Repo id / root for the on-disk aggregated dataset produced during the run.
    dataset_repo_id: str = "dagger_aggregated"
    dataset_root: Path | None = None

    # Explicit consent to execute remote code from the Hub (required for hub environments).
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        # HACK: parse the CLI again here to resolve the pretrained paths of both policies. This mirrors
        # `EvalPipelineConfig.__post_init__` but resolves two policy fields instead of one.
        self.policy = self._load_policy_from_path("policy")
        self.expert_policy = self._load_policy_from_path("expert_policy")

        if self.policy is None:
            logger.warning(
                "No student policy path provided; the student will be built from scratch (random "
                "weights) and no pretrained normalization stats will be available."
            )
        if self.expert_policy is None:
            raise ValueError(
                "DAgger requires an expert policy. Provide one with `--expert_policy.path=<repo_or_dir>`."
            )

        if not self.job_name:
            student = self.policy.type if self.policy is not None else "scratch"
            self.job_name = f"dagger_{self.env.type}_{student}"
            logger.warning(f"No job name provided, using '{self.job_name}' as job name.")

        if not self.output_dir:
            now = dt.datetime.now()
            self.output_dir = Path("outputs/dagger") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"

        if self.dataset_root is None:
            self.dataset_root = Path(self.output_dir) / "aggregated_dataset"

    @staticmethod
    def _load_policy_from_path(field_name: str) -> PreTrainedConfig | None:
        policy_path = parser.get_path_arg(field_name)
        if not policy_path:
            return None
        yaml_overrides = parser.get_yaml_overrides(field_name)
        cli_overrides = parser.get_cli_overrides(field_name) or []
        cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=yaml_overrides + cli_overrides)
        cfg.pretrained_path = Path(policy_path)
        return cfg

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Enables `--policy.path=...` and `--expert_policy.path=...` parsing."""
        return ["policy", "expert_policy"]
