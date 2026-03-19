#!/usr/bin/env python

# Copyright 2025 Nur Muhammad Mahi Shafiullah,
# and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("ditflow_mt")
@dataclass
class DiTFlowMTConfig(PreTrainedConfig):
    """Configuration class for multi-task DiTFlowPolicy."""

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Language backbone.
    tokenizer_max_length: int = 48
    language_model_name: str = "openai/clip-vit-base-patch32"
    freeze_language_pretrained: bool = True

    # Vision backbone.
    vit_name: str = 'facebook/dinov2-base'

    # Diffusion Transformer (DiT) parameters.
    frequency_embedding_dim: int = 256
    hidden_dim: int = 512
    num_blocks: int = 6
    num_heads: int = 16
    dropout: float = 0.1
    dim_feedforward: int = 4096
    activation: str = "gelu"

    # Noise scheduler.
    training_noise_sampling: str = (
        "uniform"  # "uniform" or "beta", from pi0 https://www.physicalintelligence.company/download/pi0.pdf
    )
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = 100

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        if self.training_noise_sampling not in ("uniform", "beta"):
            raise ValueError(
                f"`training_noise_sampling` must be either 'uniform' or 'beta'. Got {self.training_noise_sampling}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
