#!/usr/bin/env python

# Copyright 2025 Nur Muhammad Mahi Shafiullah,
# and The HuggingFace Inc. team. All rights reserved.
#
# Heavy inspiration taken from:
# * Multi-Task DiT by Bryson Jones
# * DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT
# * DiT Policy by Dasari et. al.: https://github.com/sudeepdasari/dit-policy
# * RDT by Meta AI: cross-attention conditioning
# * MeanFlow by Gao et. al.: mean flow objective

"""DiT (Diffusion Transformer) Policy

Transformer-based policy supporting diffusion, flow matching, and mean flow objectives
for robot learning with optional text/vision conditioning and cross-attention.

Architecture follows multi_task_dit design patterns with extensions for:
- Multiple vision encoders (DINOv2, CLIP, auto-detected from model name)
- Optional language conditioning (can be disabled)
- Cross-attention image conditioning (RDT-style)
- Mean flow objective with JVP-based training
"""

import math
from collections import deque
from functools import partial
from typing import TYPE_CHECKING

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from lerobot.utils.import_utils import _diffusers_available, _transformers_available, require_package

from .configuration_dit import DiTConfig

# Conditional imports for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel, CLIPTextModel, CLIPVisionModel
else:
    AutoModel = None
    CLIPTextModel = None
    CLIPVisionModel = None

if TYPE_CHECKING or _diffusers_available:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
else:
    DDIMScheduler = None
    DDPMScheduler = None

from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues


# =============================================================================
# Vision Encoders
# =============================================================================


class CLIPVisionEncoder(nn.Module):
    """CLIP vision encoder using the CLS token for global image representation."""

    def __init__(self, model_name: str):
        super().__init__()
        require_package("transformers", extra="multi_task_dit")
        self.model_name = model_name
        self.model = CLIPVisionModel.from_pretrained(self.model_name)
        self.num_non_spatial_tokens = 1
        self.embed_dim = self.model.config.hidden_size

    def forward(self, x: Tensor) -> Tensor:
        """Encode RGB image to CLS token."""
        outputs = self.model(pixel_values=x, output_hidden_states=False)
        cls_token = outputs.last_hidden_state[:, 0]
        b, embed_dim = cls_token.shape
        return cls_token.reshape(b, embed_dim, 1, 1)

    def get_output_shape(self) -> tuple:
        return (self.embed_dim, 1, 1)


class DINOVisionEncoder(nn.Module):
    """DINO-family vision encoder using the CLS token for global image representation.

    Uses AutoModel so it works with any DINO checkpoint exposed via transformers:
    dinov2-{base,large}, dinov2-with-registers-*, dinov3-vit{s,b,l,h,7b}16-pretrain-*,
    and dinov3-convnext-{tiny,small,base,large}-pretrain-*.
    """

    def __init__(self, model_name: str):
        super().__init__()
        require_package("transformers", extra="multi_task_dit")
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.requires_grad_(False)
        self.model.eval()
        cfg = self.model.config
        if hasattr(cfg, "hidden_size"):
            self.embed_dim = cfg.hidden_size
        elif hasattr(cfg, "hidden_sizes"):
            # DINOv3 ConvNeXt: pooler_output width = last stage channels
            self.embed_dim = cfg.hidden_sizes[-1]
        else:
            raise AttributeError(
                f"Cannot infer embed_dim from {type(cfg).__name__}: "
                "no `hidden_size` or `hidden_sizes` attribute."
            )

    def forward(self, x: Tensor) -> Tensor:
        """Encode RGB image to CLS token."""
        outputs = self.model(x)
        cls_token = outputs.pooler_output  # (B, D)
        b, embed_dim = cls_token.shape
        return cls_token.reshape(b, embed_dim, 1, 1)

    def get_output_shape(self) -> tuple:
        return (self.embed_dim, 1, 1)


def _build_vision_encoder(model_name: str) -> nn.Module:
    """Auto-detect and build the appropriate vision encoder from model name."""
    name_lower = model_name.lower()
    if "clip" in name_lower:
        return CLIPVisionEncoder(model_name=model_name)
    elif "dino" in name_lower:
        return DINOVisionEncoder(model_name=model_name)
    else:
        # Fallback to AutoModel-based encoder
        return DINOVisionEncoder(model_name=model_name)


# =============================================================================
# Text Encoder
# =============================================================================


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder with frozen weights and a learnable projection layer.

    Accepts pre-tokenized inputs (input_ids and attention_mask) from the processor pipeline.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", projection_dim: int = 512):
        super().__init__()
        require_package("transformers", extra="multi_task_dit")
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_embed_dim = self.text_encoder.config.hidden_size
        self.projection = nn.Linear(self.text_embed_dim, projection_dim)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode pre-tokenized text to feature vectors."""
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            clip_features = outputs.pooler_output

        return self.projection(clip_features)


# =============================================================================
# Observation Encoder
# =============================================================================


class ObservationEncoder(nn.Module):
    """Handles all observation processing for the conditioning vector.

    Follows multi_task_dit's ObservationEncoder pattern with extensions:
    - Auto-detected vision encoder (CLIP, DINOv2, etc.)
    - Optional language conditioning
    - Optional cross-attention image conditioning
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self._setup_preprocessing(config)

        # Vision encoder setup
        if config.image_features:
            self.num_cameras = len(config.image_features)
            self.camera_names = list(config.image_features.keys())

            if config.use_separate_rgb_encoder_per_camera:
                self.vision_encoders = nn.ModuleList(
                    [_build_vision_encoder(config.vision_encoder_name) for _ in self.camera_names]
                )
                self.vision_encoder = None
            else:
                self.vision_encoder = _build_vision_encoder(config.vision_encoder_name)
                self.vision_encoders = None

            # Freeze vision encoder if configured
            if config.freeze_vision_encoder:
                encoders = self.vision_encoders if self.vision_encoders is not None else [self.vision_encoder]
                for enc in encoders:
                    enc.requires_grad_(False)
                    enc.eval()
        else:
            self.vision_encoder = None
            self.vision_encoders = None
            self.camera_names = []
            self.num_cameras = 0

        # State dimension
        if hasattr(config, "robot_state_feature") and config.robot_state_feature:
            self.robot_state_dim = config.robot_state_feature.shape[0]
        else:
            self.robot_state_dim = 0

        # Text encoder setup (conditional)
        if config.use_language:
            self.text_dim = config.hidden_dim
            self.text_encoder = CLIPTextEncoder(
                model_name=config.text_encoder_name, projection_dim=self.text_dim
            )
        else:
            self.text_dim = 0
            self.text_encoder = None

        # Image projection for cross-attention mode
        if config.image_conditioning == "cross_attention" and config.image_features:
            encoder_to_check = self.vision_encoder or next(iter(self.vision_encoders))
            c, h, w = encoder_to_check.get_output_shape()
            self.image_proj = nn.Linear(c * h * w, config.hidden_dim)
            if config.freeze_vision_proj:
                self.image_proj.requires_grad_(False)

        self._setup_vector_output()

    def _setup_preprocessing(self, config: DiTConfig):
        if config.image_resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(
                size=config.image_resize_shape,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            self.do_resize = False

        if config.image_crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.image_crop_shape)
            if config.image_crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.image_crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

    def _apply_preprocessing(self, images: Tensor) -> Tensor:
        if self.do_resize:
            images = self.resize(images)
        if self.do_crop:
            images = self.maybe_random_crop(images) if self.training else self.center_crop(images)
        return images

    def _setup_vector_output(self):
        total_dim = 0

        # Vision features: included in conditioning_dim only for concat mode
        if (self.vision_encoder is not None or self.vision_encoders is not None) and \
                self.config.image_conditioning == "concat":
            encoder_to_check = self.vision_encoder or next(iter(self.vision_encoders))
            c, h, w = encoder_to_check.get_output_shape()
            spatial_feature_dim = c * h * w
            total_dim += spatial_feature_dim * self.num_cameras

        total_dim += self.robot_state_dim
        total_dim += self.text_dim

        self.conditioning_dim = total_dim * self.config.n_obs_steps

    def encode(self, batch: dict) -> Tensor | tuple[Tensor, Tensor]:
        """Encode observations to conditioning vector.

        Returns:
            If image_conditioning=="concat": Tensor of shape (B, conditioning_dim)
            If image_conditioning=="cross_attention": tuple of
                (conditioning_vec: (B, cond_dim), cross_cond: (B, n_img_tokens, hidden_dim))
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        conditioning_feats = []

        # State features — flatten temporal dim so all conditioning_feats are (B, flat)
        conditioning_feats.append(batch[OBS_STATE].flatten(start_dim=1))

        # Vision features
        img_features_for_cross = None
        if self.vision_encoder is not None or self.vision_encoders is not None:
            images = batch[OBS_IMAGES]
            if len(images.shape) == 5:
                images = images.unsqueeze(1)

            if self.config.use_separate_rgb_encoder_per_camera:
                camera_features = []
                for cam_idx in range(self.num_cameras):
                    cam_images = images[:, :, cam_idx]
                    cam_images_flat = einops.rearrange(cam_images, "b s c h w -> (b s) c h w")
                    cam_images_flat = self._apply_preprocessing(cam_images_flat)
                    with torch.no_grad() if self.config.freeze_vision_encoder else torch.enable_grad():
                        cam_features = self.vision_encoders[cam_idx](cam_images_flat)
                    cam_visual_features = cam_features.flatten(start_dim=1)
                    cam_features_reshaped = einops.rearrange(
                        cam_visual_features, "(b s) f -> b s f", b=batch_size, s=n_obs_steps
                    )
                    camera_features.append(cam_features_reshaped)
                img_features = torch.cat(camera_features, dim=-1)
            else:
                images_flat = einops.rearrange(images, "b s n c h w -> (b s n) c h w")
                images_flat = self._apply_preprocessing(images_flat)
                with torch.no_grad() if self.config.freeze_vision_encoder else torch.enable_grad():
                    visual_features = self.vision_encoder(images_flat).flatten(start_dim=1)
                img_features = einops.rearrange(
                    visual_features, "(b s n) f -> b s (n f)",
                    b=batch_size, s=n_obs_steps, n=self.num_cameras
                )

            if self.config.image_conditioning == "concat":
                conditioning_feats.append(img_features.flatten(start_dim=1))
            else:
                # Cross-attention: project image features to hidden_dim tokens
                # img_features: (B, n_obs_steps, n_cameras * feat_dim)
                # We want per-camera-per-step tokens: (B, n_obs_steps * n_cameras, hidden_dim)
                if self.config.use_separate_rgb_encoder_per_camera:
                    # camera_features is a list of (B, n_obs_steps, feat_dim)
                    all_cam = torch.stack(camera_features, dim=2)  # (B, S, N, F)
                    all_cam = einops.rearrange(all_cam, "b s n f -> b (s n) f")
                else:
                    all_cam = einops.rearrange(
                        visual_features, "(b s n) f -> b (s n) f",
                        b=batch_size, s=n_obs_steps, n=self.num_cameras
                    )
                img_features_for_cross = self.image_proj(all_cam)  # (B, S*N, hidden_dim)

        # Text features (conditional)
        if self.text_encoder is not None and OBS_LANGUAGE_TOKENS in batch:
            input_ids = batch[OBS_LANGUAGE_TOKENS]
            attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
            text_features = self.text_encoder(input_ids, attention_mask)
            text_features = text_features.unsqueeze(1).expand(-1, n_obs_steps, -1)
            conditioning_feats.append(text_features.flatten(start_dim=1))

        conditioning_vec = torch.cat(conditioning_feats, dim=-1)  # (B, conditioning_dim)

        if self.config.image_conditioning == "cross_attention" and img_features_for_cross is not None:
            return conditioning_vec, img_features_for_cross
        return conditioning_vec


# =============================================================================
# Transformer Components
# =============================================================================


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Modulate input with shift and scale for AdaLN-Zero."""
    return x * (1 + scale) + shift


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformers."""

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("_sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.")

        cos = self._cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self._sin_cached[:, :, :seq_len, :].to(q.dtype)

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        return q_rotated, k_rotated


class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim, max_seq_len=max_seq_len, base=rope_base
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape  # noqa: N806

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) and self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.hidden_size)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """DiT-style transformer block with AdaLN-Zero."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_features: int = 128,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.use_rope = use_rope

        if use_rope:
            self.attn = RoPEAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
            )
        else:
            self.multihead_attn = nn.MultiheadAttention(
                hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
            )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(num_features, 6 * hidden_size, bias=True))

    def forward(self, x: Tensor, features: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            features
        ).chunk(6, dim=1)

        attn_input = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))

        if self.use_rope:
            attn_out = self.attn(attn_input)
        else:
            attn_out, _ = self.multihead_attn(attn_input, attn_input, attn_input)

        x = x + gate_msa.unsqueeze(1) * attn_out

        mlp_input = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        mlp_out = self.mlp(mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class CrossAttentionTransformerBlock(nn.Module):
    """DiT-style block with AdaLN-Zero self-attention + cross-attention on image tokens.

    Following RDT's pattern: self-attention → cross-attention → FFN.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_features: int = 128,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.use_rope = use_rope

        # Self-attention
        if use_rope:
            self.attn = RoPEAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
            )
        else:
            self.multihead_attn = nn.MultiheadAttention(
                hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
            )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Cross-attention for image conditioning
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)

        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # AdaLN modulation for self-attention + MLP (6 params)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(num_features, 6 * hidden_size, bias=True))

    def forward(self, x: Tensor, features: Tensor, cross_cond: Tensor | None = None) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            features
        ).chunk(6, dim=1)

        # Self-attention with AdaLN
        attn_input = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        if self.use_rope:
            attn_out = self.attn(attn_input)
        else:
            attn_out, _ = self.multihead_attn(attn_input, attn_input, attn_input)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Cross-attention on image tokens
        if cross_cond is not None:
            x_norm = self.norm_cross(x)
            cross_out, _ = self.cross_attn(x_norm, cross_cond, cross_cond)
            x = x + cross_out

        # MLP with AdaLN
        mlp_input = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        mlp_out = self.mlp(mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


# =============================================================================
# Diffusion Transformer (Noise Predictor)
# =============================================================================


class DiffusionTransformer(nn.Module):
    """Transformer-based diffusion noise prediction model.

    Supports both standard AdaLN-Zero conditioning and cross-attention image conditioning.
    For mean flow, includes a second time embedding network.
    """

    def __init__(self, config: DiTConfig, conditioning_dim: int):
        super().__init__()
        self.config = config
        self.conditioning_dim = conditioning_dim

        self.action_dim = config.action_feature.shape[0]
        self.horizon = config.horizon
        self.hidden_size = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        # Time embedding
        self.timestep_embed_dim = config.timestep_embed_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.timestep_embed_dim),
            nn.Linear(self.timestep_embed_dim, 2 * self.timestep_embed_dim),
            nn.GELU(),
            nn.Linear(2 * self.timestep_embed_dim, self.timestep_embed_dim),
            nn.GELU(),
        )

        # Second time embedding for mean flow (t - r)
        if config.is_mean_flow:
            self.time_mlp_r = nn.Sequential(
                SinusoidalPosEmb(self.timestep_embed_dim),
                nn.Linear(self.timestep_embed_dim, 2 * self.timestep_embed_dim),
                nn.GELU(),
                nn.Linear(2 * self.timestep_embed_dim, self.timestep_embed_dim),
                nn.GELU(),
            )

        self.cond_dim = self.timestep_embed_dim + conditioning_dim
        self.input_proj = nn.Linear(self.action_dim, self.hidden_size)

        if config.use_positional_encoding:
            self.pos_embedding = nn.Parameter(
                torch.empty(1, self.horizon, self.hidden_size).normal_(std=0.02)
            )
        else:
            self.pos_embedding = None

        # Select block type based on image conditioning mode
        use_cross_attention = config.image_conditioning == "cross_attention"
        BlockClass = CrossAttentionTransformerBlock if use_cross_attention else TransformerBlock

        self.transformer_blocks = nn.ModuleList(
            [
                BlockClass(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    num_features=self.cond_dim,
                    dropout=self.dropout,
                    use_rope=self.use_rope,
                    max_seq_len=self.horizon,
                    rope_base=config.rope_base,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_proj = nn.Linear(self.hidden_size, self.action_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        conditioning_vec: Tensor,
        cross_cond: Tensor | None = None,
        timestep_r: Tensor | None = None,
    ) -> Tensor:
        _, seq_len, _ = x.shape

        # Time embedding
        timestep_features = self.time_mlp(timestep)
        if timestep_r is not None and hasattr(self, "time_mlp_r"):
            timestep_features = timestep_features + self.time_mlp_r(timestep - timestep_r)

        cond_features = torch.cat([timestep_features, conditioning_vec], dim=-1)

        hidden_seq = self.input_proj(x)

        if self.pos_embedding is not None:
            hidden_seq = hidden_seq + self.pos_embedding[:, :seq_len, :]

        for block in self.transformer_blocks:
            if isinstance(block, CrossAttentionTransformerBlock):
                hidden_seq = block(hidden_seq, cond_features, cross_cond=cross_cond)
            else:
                hidden_seq = block(hidden_seq, cond_features)

        return self.output_proj(hidden_seq)


# =============================================================================
# Objectives
# =============================================================================


class DiffusionObjective(nn.Module):
    """Standard diffusion (DDPM/DDIM) objective implementation."""

    def __init__(self, config: DiTConfig, action_dim: int, horizon: int, do_mask_loss_for_padding: bool = False):
        super().__init__()
        require_package("diffusers", extra="multi_task_dit")
        self.config = config
        self.action_dim = action_dim
        self.horizon = horizon
        self.do_mask_loss_for_padding = do_mask_loss_for_padding

        scheduler_kwargs = {
            "num_train_timesteps": config.num_train_timesteps,
            "beta_start": config.beta_start,
            "beta_end": config.beta_end,
            "beta_schedule": config.beta_schedule,
            "clip_sample": config.clip_sample,
            "clip_sample_range": config.clip_sample_range,
            "prediction_type": config.prediction_type,
        }

        if config.noise_scheduler_type == "DDPM":
            self.noise_scheduler: DDPMScheduler | DDIMScheduler = DDPMScheduler(**scheduler_kwargs)
        elif config.noise_scheduler_type == "DDIM":
            self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported noise scheduler type {config.noise_scheduler_type}")

        self.num_inference_steps = (
            config.num_inference_steps
            if config.num_inference_steps is not None
            else self.noise_scheduler.config.num_train_timesteps
        )

    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
        conditioning_vec: Tensor,
        cross_cond: Tensor | None = None,
    ) -> Tensor:
        clean_actions = batch[ACTION]
        noise = torch.randn_like(clean_actions)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(clean_actions.shape[0],),
            device=clean_actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(clean_actions, noise, timesteps)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = clean_actions
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        predicted = model(noisy_actions, timesteps, conditioning_vec=conditioning_vec, cross_cond=cross_cond)
        loss = F.mse_loss(predicted, target, reduction="none")

        if self.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid_actions = ~batch["action_is_pad"]
            loss = loss * valid_actions.unsqueeze(-1)

        return loss.mean()

    def conditional_sample(
        self,
        model: nn.Module,
        batch_size: int,
        conditioning_vec: Tensor,
        cross_cond: Tensor | None = None,
    ) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        sample = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = model(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                conditioning_vec=conditioning_vec,
                cross_cond=cross_cond,
            )
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample

        return sample


class FlowMatchingObjective(nn.Module):
    """Flow matching objective: trains a model to predict velocity fields."""

    def __init__(self, config: DiTConfig, action_dim: int, horizon: int, do_mask_loss_for_padding: bool = False):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.horizon = horizon
        self.do_mask_loss_for_padding = do_mask_loss_for_padding

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        if self.config.timestep_sampling_strategy == "uniform":
            return torch.rand(batch_size, device=device)
        elif self.config.timestep_sampling_strategy == "beta":
            beta_dist = torch.distributions.Beta(
                self.config.timestep_sampling_alpha, self.config.timestep_sampling_beta
            )
            u = beta_dist.sample((batch_size,)).to(device)
            return self.config.timestep_sampling_s * (1.0 - u)
        else:
            raise ValueError(f"Unknown timestep strategy: {self.config.timestep_sampling_strategy}")

    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
        conditioning_vec: Tensor,
        cross_cond: Tensor | None = None,
    ) -> Tensor:
        data = batch[ACTION]
        batch_size = data.shape[0]
        device = data.device

        noise = torch.randn_like(data)
        t = self._sample_timesteps(batch_size, device)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * data + (1 - (1 - self.config.sigma_min) * t_expanded) * noise

        target_velocity = data - (1 - self.config.sigma_min) * noise
        predicted_velocity = model(x_t, t, conditioning_vec=conditioning_vec, cross_cond=cross_cond)
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")

        if self.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid_mask = ~batch["action_is_pad"]
            loss = loss * valid_mask.unsqueeze(-1)

        return loss.mean()

    def conditional_sample(
        self,
        model: nn.Module,
        batch_size: int,
        conditioning_vec: Tensor,
        cross_cond: Tensor | None = None,
    ) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        x = torch.randn((batch_size, self.horizon, self.action_dim), dtype=dtype, device=device)

        num_steps = self.config.num_integration_steps
        time_grid = torch.linspace(0, 1, num_steps + 1, device=device)

        if self.config.integration_method == "euler":
            x = self._euler_integrate(model, x, time_grid, conditioning_vec, cross_cond)
        elif self.config.integration_method == "rk4":
            x = self._rk4_integrate(model, x, time_grid, conditioning_vec, cross_cond)
        else:
            raise ValueError(f"Unknown integration method: {self.config.integration_method}")

        return x

    def _euler_integrate(
        self, model: nn.Module, x_init: Tensor, time_grid: Tensor,
        conditioning_vec: Tensor, cross_cond: Tensor | None = None,
    ) -> Tensor:
        x = x_init
        for i in range(len(time_grid) - 1):
            t_scalar = time_grid[i].item()
            dt = (time_grid[i + 1] - time_grid[i]).item()
            t_batch = torch.full((x.shape[0],), t_scalar, dtype=x.dtype, device=x.device)
            with torch.no_grad():
                velocity = model(x, t_batch, conditioning_vec=conditioning_vec, cross_cond=cross_cond)
            x = x + dt * velocity
        return x

    def _rk4_integrate(
        self, model: nn.Module, x_init: Tensor, time_grid: Tensor,
        conditioning_vec: Tensor, cross_cond: Tensor | None = None,
    ) -> Tensor:
        x = x_init

        def dynamics(x_val: Tensor, t_scalar: float) -> Tensor:
            t_batch = torch.full((x_val.shape[0],), t_scalar, dtype=x_val.dtype, device=x_val.device)
            with torch.no_grad():
                return model(x_val, t_batch, conditioning_vec=conditioning_vec, cross_cond=cross_cond)

        for i in range(len(time_grid) - 1):
            t = time_grid[i].item()
            dt = (time_grid[i + 1] - time_grid[i]).item()

            k1 = dynamics(x, t)
            k2 = dynamics(x + dt * k1 / 2, t + dt / 2)
            k3 = dynamics(x + dt * k2 / 2, t + dt / 2)
            k4 = dynamics(x + dt * k3, t + dt)

            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return x


class MeanFlowObjective(nn.Module):
    """Mean flow objective: trains a model using JVP-based mean velocity targets.

    Reference: MeanFlow by Gao et al.
    """

    def __init__(self, config: DiTConfig, action_dim: int, horizon: int, do_mask_loss_for_padding: bool = False):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.horizon = horizon
        self.do_mask_loss_for_padding = do_mask_loss_for_padding

        # Set up JVP function
        if config.use_autograd_functional_jvp:
            import torch.autograd.functional
            self.jvp_func = partial(torch.autograd.functional.jvp, create_graph=True)
        else:
            import torch.func
            self.jvp_func = torch.func.jvp

    def _sample_time_distribution(self, batch_size: int, num_time_parameters: int) -> np.ndarray:
        """Sample from the configured time distribution."""
        if self.config.time_distribution == "uniform":
            samples = np.random.rand(batch_size, num_time_parameters).astype(np.float32)
        elif self.config.time_distribution == "logit_normal":
            normal_samples = (
                np.random.randn(batch_size, num_time_parameters).astype(np.float32)
                * self.config.log_norm_sigma
                + self.config.log_norm_mu
            )
            samples = 1 / (1 + np.exp(-normal_samples))  # sigmoid
        else:
            raise ValueError(f"Unknown time distribution: {self.config.time_distribution}")
        return samples

    def _sample_t_and_r(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Sample paired (t, r) timesteps for mean flow training."""
        samples = self._sample_time_distribution(batch_size, num_time_parameters=2)

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # Mix in pure flow samples based on flow_ratio
        num_selected = int(self.config.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def _sample_t(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample a single timestep for standard flow loss."""
        samples = self._sample_time_distribution(batch_size, num_time_parameters=1)
        return torch.tensor(samples[:, 0], device=device)

    @staticmethod
    def adaptive_l2_loss(u: Tensor, u_target: Tensor, gamma: float = 0.5, c: float = 1e-3) -> Tensor:
        """Adaptive L2 loss as in AdaLFD paper."""
        error = u - u_target
        delta_sq = torch.mean(error**2, dim=tuple(range(1, error.ndim)))
        p = 1.0 - gamma
        w = 1.0 / (delta_sq + c).pow(p)
        return (w.detach() * delta_sq).mean()

    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
        conditioning_vec: Tensor,
        cross_cond: Tensor | None = None,
    ) -> Tensor:
        trajectory = batch[ACTION]
        batch_size = trajectory.shape[0]
        device = trajectory.device

        # Sample noise
        noise = torch.randn_like(trajectory)

        # Sample (t, r) timesteps
        t, r = self._sample_t_and_r(batch_size, device)
        t_ = t[:, None, None]
        r_ = r[:, None, None]

        # Create noisy trajectory
        noisy_trajectory = (1 - t_) * trajectory + t_ * noise

        # Target velocity
        v = noise - trajectory

        # Classifier-free guidance
        if self.config.use_classifier_free_guidance:
            conditional_mask = torch.rand_like(t) < self.config.cfg_prob
            unconditional_cond = torch.zeros_like(conditioning_vec)
            conditioning_vec_cfg = torch.where(
                conditional_mask.unsqueeze(-1), unconditional_cond, conditioning_vec
            )
            with torch.no_grad():
                u_uncond = model(
                    noisy_trajectory, t, conditioning_vec=unconditional_cond,
                    cross_cond=cross_cond, timestep_r=t,
                )
            v_hat = self.config.cfg_omega * v + (1 - self.config.cfg_omega) * u_uncond
            v_hat = torch.where(conditional_mask.unsqueeze(-1).unsqueeze(-1), v, v_hat)
            cond_for_jvp = conditioning_vec_cfg
        else:
            v_hat = v
            cond_for_jvp = conditioning_vec

        # JVP computation for mean target velocity.
        # torch.func.jvp requires autocast to be inside the traced function —
        # an outer autocast context causes primal/tangent dtype inconsistency at
        # backward time. Capture the outer state, disable it for the JVP call,
        # and re-enable it inside model_partial so the trace is self-consistent.
        device_type = noisy_trajectory.device.type
        amp_enabled = torch.is_autocast_enabled()
        amp_dtype = torch.get_autocast_gpu_dtype() if amp_enabled else torch.float32

        def model_partial(z_k, t_k, r_k):
            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled, dtype=amp_dtype):
                return model(
                    z_k, t_k, conditioning_vec=cond_for_jvp,
                    cross_cond=cross_cond, timestep_r=r_k,
                )

        # sdpa_kernel(MATH): flash/efficient kernels have no forward-AD rule.
        # autocast(enabled=False): disable outer autocast; model_partial re-enables it internally.
        with sdpa_kernel(SDPBackend.MATH), torch.amp.autocast(device_type=device_type, enabled=False):
            u, dudt = self.jvp_func(
                model_partial,
                (noisy_trajectory, t, r),
                (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            )

        # Mean flow target
        u_target = v_hat - (t_ - r_) * dudt

        # Compute loss
        if self.config.use_adaptive_loss:
            loss = self.adaptive_l2_loss(u, u_target.detach())
        else:
            loss = F.mse_loss(u, u_target.detach(), reduction="none")

        if self.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid_mask = ~batch["action_is_pad"]
            loss = loss * valid_mask.unsqueeze(-1)

        return loss.mean()

    def conditional_sample(
        self,
        model: nn.Module,
        batch_size: int,
        conditioning_vec: Tensor,
        cross_cond: Tensor | None = None,
    ) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        x = torch.randn((batch_size, self.horizon, self.action_dim), dtype=dtype, device=device)

        if not self.config.do_multi_step_sampling:
            # One-step inference
            t = torch.ones(batch_size, device=device)
            r = torch.zeros(batch_size, device=device)
            with torch.no_grad():
                u = model(x, t, conditioning_vec=conditioning_vec, cross_cond=cross_cond, timestep_r=r)
            x = x - u
        else:
            # Multi-step Euler integration
            timesteps = self.config.inference_timesteps
            dt = 1.0 / timesteps
            t_vals = torch.linspace(0.0, 1.0, timesteps + 1, device=device)

            for k in range(timesteps):
                t = torch.full((batch_size,), t_vals[k], device=device)
                r = torch.full((batch_size,), t_vals[k + 1], device=device)
                with torch.no_grad():
                    u = model(x, t, conditioning_vec=conditioning_vec, cross_cond=cross_cond, timestep_r=r)
                x = x - dt * u

        if self.config.clip_sample:
            x = torch.clamp(x, -self.config.clip_sample_range, self.config.clip_sample_range)

        return x


# =============================================================================
# Policy
# =============================================================================


class DiTPolicy(PreTrainedPolicy):
    """DiT Policy for robot learning via diffusion, flow matching, or mean flow.

    Follows multi_task_dit's MultiTaskDiTPolicy design pattern.
    """

    config_class = DiTConfig
    name = "dit"

    def __init__(self, config: DiTConfig, **kwargs):
        require_package("transformers", extra="multi_task_dit")
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None

        self.observation_encoder = ObservationEncoder(config)
        conditioning_dim = self.observation_encoder.conditioning_dim
        self.noise_predictor = DiffusionTransformer(config, conditioning_dim=conditioning_dim)

        action_dim = config.action_feature.shape[0]
        horizon = config.horizon

        if config.is_diffusion:
            require_package("diffusers", extra="multi_task_dit")
            self.objective = DiffusionObjective(
                config,
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
            )
        elif config.is_flow_matching:
            self.objective = FlowMatchingObjective(
                config,
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
            )
        elif config.is_mean_flow:
            self.objective = MeanFlowObjective(
                config,
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
            )
        else:
            raise ValueError(f"Unsupported objective: {config.objective}")

        self.reset()

    def get_optim_params(self) -> list:
        """Returns parameter groups with different learning rates for vision vs non-vision parameters."""
        non_vision_params = []
        vision_encoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "observation_encoder.vision_encoder" in name:
                vision_encoder_params.append(param)
            else:
                non_vision_params.append(param)

        return [
            {"params": non_vision_params},
            {
                "params": vision_encoder_params,
                "lr": self.config.optimizer_lr * self.config.vision_encoder_lr_multiplier,
            },
        ]

    def _generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        encode_result = self.observation_encoder.encode(batch)

        if isinstance(encode_result, tuple):
            conditioning_vec, cross_cond = encode_result
        else:
            conditioning_vec = encode_result
            cross_cond = None

        actions = self.objective.conditional_sample(
            self.noise_predictor, batch_size, conditioning_vec, cross_cond=cross_cond
        )

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]
        return actions

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        for k in batch:
            if k in self._queues:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        actions = self._generate_actions(batch)
        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare batch by stacking image features and concatenating state features if needed."""
        batch = dict(batch)
        if self.config.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # Concatenate multiple state features into OBS_STATE if needed
        state_keys = list(self.config.state_features.keys())
        if len(state_keys) > 1 or (len(state_keys) == 1 and state_keys[0] != OBS_STATE):
            state_tensors = [batch[k] for k in state_keys]
            target_ndim = max(t.dim() for t in state_tensors)
            state_tensors = [t.unsqueeze(-1) if t.dim() == target_ndim - 1 else t for t in state_tensors]
            batch[OBS_STATE] = torch.cat(state_tensors, dim=-1)

        return batch

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        batch = self._prepare_batch(batch)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Run the batch through the model and compute the loss for training."""
        batch = self._prepare_batch(batch)

        encode_result = self.observation_encoder.encode(batch)

        if isinstance(encode_result, tuple):
            conditioning_vec, cross_cond = encode_result
        else:
            conditioning_vec = encode_result
            cross_cond = None

        loss = self.objective.compute_loss(
            self.noise_predictor, batch, conditioning_vec, cross_cond=cross_cond
        )

        return loss, None
