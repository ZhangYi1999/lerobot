# Copyright 2025 Nur Muhammad Mahi Shafiullah,
# and The HuggingFace Inc. team. All rights reserved.
# Heavy inspiration taken from
# * DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# * DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT
# * DiT Policy by Dasari et. al. : https://github.com/sudeepdasari/dit-policy

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel

from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE, ACTION, OBS_IMAGES
from lerobot.policies.dit.configuration_dit import DiTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)

USE_STATE_PROJ = False
NAMING_AS_MLP = True

def _get_activation_fn(activation: str):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)



class LanguageEncoder(nn.Module):
    """
    Language Encoder using pretrained CLIP "Learning Transferable Visual Models From Natural Language Supervision"
    (paper: https://arxiv.org/pdf/2103.00020)
    """

    def __init__(self, config:DiTConfig):
        super().__init__()

        self.config = config

        self.tokenizer = CLIPTokenizer.from_pretrained(config.language_model_name)
        self.clip_model = CLIPTextModel.from_pretrained(config.language_model_name)
        self.cache = {}

        # Get the hidden size of CLIP model
        self.hidden_size = self.clip_model.config.hidden_size

        # Freeze the base model if specified
        if config.freeze_language_pretrained:
            self.clip_model.requires_grad_(False)
            self.clip_model.eval()

    def forward(self, texts):
        """
        Encodes input text into embeddings and projects to specified output dimension.

        Args:
            texts (list[str]): List of text strings to be encoded (batch size B).

        Returns:
            torch.Tensor: The projected text embeddings of shape (B, output_dim).
        """
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self.cache:
                cached_embeddings.append(self.cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Process uncached texts
        if uncached_texts:
            # Tokenize the input texts
            inputs = self.tokenizer(
                uncached_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length
            )

            # Move inputs to the same device as the model
            device = self.config.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings from CLIP text model
            with torch.set_grad_enabled(not self.clip_model.training):
                outputs = self.clip_model(**inputs)

            # Get the EOS token embeddings
            uncached_embeddings = outputs.pooler_output

            # Update cache
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                self.cache[text] = embedding.detach().cpu()  # Store in CPU to save GPU memory

        # Combine cached and uncached embeddings in the original order
        all_embeddings = [None] * len(texts)
        # Process uncached texts
        if uncached_texts:
            for i, emb in zip(uncached_indices, uncached_embeddings):
                all_embeddings[i] = emb
        for i, text in enumerate(texts):
            if text in self.cache and all_embeddings[i] is None:
                # Move cached embedding to same device as model
                all_embeddings[i] = self.cache[text].to(self.config.device)

        # Stack all embeddings into a single tensor
        return torch.stack(all_embeddings)


class DINOv2Encoder(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self._model = AutoModel.from_pretrained(config.vit_name)
        self._model.to(config.device)
        self._model.requires_grad_(False) # hack
        self._model.eval() # hack

        self.hidden_size = self._model.config.hidden_size

        self.crop_shape = config.crop_shape
        self.crop_ratio = config.crop_ratio
        self.crop_is_random = config.crop_is_random

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resolve crop size: crop_shape > crop_ratio > no crop
        if self.crop_shape is not None:
            crop_H, crop_W = self.crop_shape
        elif self.crop_ratio is not None:
            _, _, H, W = x.shape
            crop_H, crop_W = int(H * self.crop_ratio), int(W * self.crop_ratio)
        else:
            crop_H = None

        if crop_H is not None:
            if self.training and self.crop_is_random:
                _, _, H, W = x.shape
                x = torchvision.transforms.functional.crop(
                    x,
                    top=torch.randint(0, H - crop_H + 1, (1,)).item(),
                    left=torch.randint(0, W - crop_W + 1, (1,)).item(),
                    height=crop_H, width=crop_W,
                )
            else:
                x = torchvision.transforms.functional.center_crop(x, [crop_H, crop_W])

        outputs = self._model(x)
        cls_token = outputs.pooler_output  # (B, 768)
        return cls_token


class _TimeNetwork(nn.Module):
    def __init__(self, frequency_embedding_dim, hidden_dim, learnable_w=False, max_period=1000):
        assert frequency_embedding_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(frequency_embedding_dim // 2)
        super().__init__()

        w = np.log(max_period) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t):
        assert len(t.shape) == 1, "assumes 1d input timestep array"
        t = t[:, None] * self.w[None]
        t = torch.cat((torch.cos(t), torch.sin(t)), dim=1)
        return self.out_net(t)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * (1 + self.scale(c)[None]) + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class MLP(nn.Module):
    def __init__(self, d_model=256, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.dropout3(x)

        return x


class _DiTDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=6, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        if NAMING_AS_MLP:
            self.mlp = MLP(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.activation = _get_activation_fn(activation)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)

        # create modulation layers
        self.attn_modulate = _ShiftScaleMod(d_model)
        self.attn_gate = _ZeroScaleMod(d_model)
        self.mlp_modulate = _ShiftScaleMod(d_model)
        self.mlp_gate = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = cond + t

        x2 = self.attn_modulate(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = x + self.attn_gate(self.dropout1(x2), cond)

        x3 = self.mlp_modulate(self.norm2(x), cond)

        if NAMING_AS_MLP:
            x3 = self.mlp(x3)
        else:
            x3 = self.activation(self.linear1(x3))
            x3 = self.dropout2(x3)
            x3 = self.linear2(x3)
            x3 = self.dropout3(x3)

        x3 = self.mlp_gate(x3, cond)
        return x + x3

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_modulate, self.attn_gate, self.mlp_modulate, self.mlp_gate):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (T, B, d_model) -> returns positional encoding (T, B, d_model)"""
        pe = self.pe[:x.shape[0]]
        return pe.expand(-1, x.shape[1], -1)


class _DiTEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = MLP(d_model, dim_feedforward, dropout, activation)

    def forward(self, src, pos_embed=None):
        q = k = _with_pos_embed(src, pos_embed)
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + src2
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.0, activation="gelu", num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            _DiTEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, pos_embed=None):
        for layer in self.layers:
            x = layer(x, pos_embed)
        return x


class _TransformerDecoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(base_module) for _ in range(num_layers)])

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, src, t, cond):
        x = src
        for layer in self.layers:
            x = layer(x, t, cond)
        return x


class _DiTNoiseNet(nn.Module):
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        cond_dim=None,
        time_dim=256,
        hidden_dim=256,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
        clip_sample=False,
        clip_sample_range=1.0,
        use_encoder=False,
        n_encoder_layers=6,
    ):
        super().__init__()
        self.ac_dim, self.ac_chunk = ac_dim, ac_chunk
        self.use_encoder = use_encoder

        # positional encoding blocks
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encoder mlps
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )

        # conditioning projection: linear or encoder
        if use_encoder:
            self.cond_proj = nn.ModuleDict({
                "encoder": _TransformerEncoder(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    num_layers=n_encoder_layers,
                ),
                "pos_enc": _PositionalEncoding(hidden_dim),
            })
        else:
            assert cond_dim is not None
            self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # decoder blocks
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        # clip the output samples
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

    def forward_enc(self, global_cond):
        """Encode token-based conditioning via transformer encoder.

        Args:
            global_cond: (B, T, hidden_dim) conditioning tokens.
        Returns:
            enc_cache: (T, B, hidden_dim) encoder output.
        """
        x = global_cond.transpose(0, 1)  # (B, T, D) -> (T, B, D)
        pos = self.cond_proj["pos_enc"](x)
        return self.cond_proj["encoder"](x, pos)

    def forward(self, noisy_actions, time, global_cond, enc_cache=None):
        if self.use_encoder:
            if enc_cache is None:
                enc_cache = self.forward_enc(global_cond)
            # Average encoder output across sequence dim
            c = torch.mean(enc_cache, dim=0)  # (T, B, D) -> (B, D)
        else:
            c = self.cond_proj(global_cond)

        time_enc = self.time_net(time)

        ac_tokens = self.ac_proj(noisy_actions)  # [B, T, adim] -> [B, T, hidden_dim]
        ac_tokens = ac_tokens.transpose(0, 1)  # [B, T, hidden_dim] -> [T, B, hidden_dim]

        # Allow variable length action chunks
        dec_in = ac_tokens + self.dec_pos[: ac_tokens.size(0)]  # [T, B, hidden_dim]

        # apply decoder
        dec_out = self.decoder(dec_in, time_enc, c)

        # apply final epsilon prediction layer
        eps_out = self.eps_out(dec_out, time_enc, c)  # [T, B, hidden_dim] -> [T, B, adim]
        return eps_out.transpose(0, 1)  # [T, B, adim] -> [B, T, adim]

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        timesteps: int = 100,
        generator: torch.Generator | None = None,
        noise_scheduler=None,
    ) -> torch.Tensor:
        batch_size = condition.shape[0] if not self.use_encoder else condition.shape[0]
        device = condition.device
        x = self.sample_noise(batch_size, device, generator)

        # Pre-compute encoder cache if using encoder
        enc_cache = None
        if self.use_encoder:
            enc_cache = self.forward_enc(condition)

        if noise_scheduler is not None:
            # DDIM path
            noise_scheduler.set_timesteps(timesteps)
            for t in noise_scheduler.timesteps:
                t_batch = torch.full(
                    (batch_size,), t, dtype=torch.long, device=device
                )
                model_output = self.forward(
                    x, t_batch, condition, enc_cache=enc_cache
                )
                x = noise_scheduler.step(
                    model_output, t, x, generator=generator
                ).prev_sample
        else:
            # Flow matching path (Euler ODE solver)
            dt = 1.0 / timesteps
            t_all = (
                torch.arange(timesteps, device=device)
                .float().unsqueeze(0).expand(batch_size, timesteps)
                / timesteps
            )
            for k in range(timesteps):
                t = t_all[:, k]
                x = x + dt * self.forward(x, t, condition, enc_cache=enc_cache)

        if self.clip_sample:
            x = torch.clamp(x, -self.clip_sample_range, self.clip_sample_range)
        return x

    def sample_noise(self, batch_size: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(batch_size, self.ac_chunk, self.ac_dim, device=device, generator=generator)


class DiTPolicy(PreTrainedPolicy):
    """
    DiT Flow Policy for visuomotor policy learning via flow matching.
    """

    config_class = DiTConfig
    name = "dit"

    def __init__(self, config: DiTConfig, **kwargs):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.dit = DiTModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.dit.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        for key in batch:
            if key in self._queues:
                batch[key] = torch.stack(list(self._queues[key]), dim=1)

        actions = self.dit.generate_actions(batch)

        return actions

    @torch.no_grad
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying flow model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The flow model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        loss = self.dit.compute_loss(batch)
        return loss, None


class DiTModel(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.noise_type = config.noise_type
        self.use_encoder = config.use_encoder

        # Build observation encoders (depending on which observations are provided).
        self.language_encoder = LanguageEncoder(config).to(self.config.device)
        self.language_embedding_projection = nn.Linear(
            self.language_encoder.hidden_size, config.hidden_dim
        )
        if config.freeze_language_proj:
            self.language_embedding_projection.requires_grad_(False)

        if self.config.image_features:
            self.pretrained_rgb_encoder = DINOv2Encoder(config)
            self.rgb_embedding_projection = nn.Linear(
                self.pretrained_rgb_encoder.hidden_size, config.hidden_dim
            )
            if config.freeze_vision_proj:
                self.rgb_embedding_projection.requires_grad_(False)

        if config.use_encoder:
            # Token-based conditioning: all features projected to hidden_dim
            self.state_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.hidden_dim
            )
            self.state_dropout = nn.Dropout(config.state_dropout)
            if config.freeze_state_proj:
                self.state_proj.requires_grad_(False)
            if self.config.env_state_feature:
                self.env_state_proj = nn.Linear(
                    self.config.env_state_feature.shape[0], config.hidden_dim
                )
            cond_dim = None  # not used when encoder is active
        else:
            # Flat conditioning (original behavior)
            if USE_STATE_PROJ:
                global_cond_dim = config.hidden_dim
                self.state_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.hidden_dim
                )
                self.state_dropout = nn.Dropout(config.state_dropout)
                if config.freeze_state_proj:
                    self.state_proj.requires_grad_(False)
            else:
                global_cond_dim = self.config.robot_state_feature.shape[0]

            if self.config.image_features:
                global_cond_dim += config.hidden_dim * len(self.config.image_features)

            if self.config.env_state_feature:
                global_cond_dim += self.config.env_state_feature.shape[0]

            language_cond_dim = config.hidden_dim
            cond_dim = language_cond_dim + global_cond_dim * config.n_obs_steps

        self.velocity_net = _DiTNoiseNet(
            ac_dim=config.action_feature.shape[0],
            ac_chunk=config.horizon,
            cond_dim=cond_dim,
            time_dim=config.frequency_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            dim_feedforward=config.dim_feedforward,
            nhead=config.num_heads,
            activation=config.activation,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            use_encoder=config.use_encoder,
            n_encoder_layers=config.n_encoder_layers,
        )

        self.num_inference_steps = config.num_inference_steps or 100

        # Noise type setup
        if config.noise_type == "flow_matching":
            self.training_noise_sampling = config.training_noise_sampling
            if config.training_noise_sampling == "uniform":
                self.noise_distribution = torch.distributions.Uniform(
                    low=0, high=1,
                )
            elif config.training_noise_sampling == "beta":
                s = 0.999
                beta_dist = torch.distributions.Beta(
                    concentration1=1.5, concentration0=1.0,
                )
                affine_transform = torch.distributions.transforms.AffineTransform(loc=s, scale=-s)
                self.noise_distribution = torch.distributions.TransformedDistribution(
                    beta_dist, [affine_transform]
                )
        elif config.noise_type == "ddim":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=config.num_train_timesteps,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                beta_schedule=config.beta_schedule,
                clip_sample=False,  # we handle clipping in _DiTNoiseNet
                prediction_type=config.prediction_type,
                set_alpha_to_one=config.set_alpha_to_one,
                steps_offset=config.steps_offset,
            )

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        if global_cond is not None:
            global_cond = global_cond.to(device=device, dtype=dtype)

        noise_scheduler = None
        if self.noise_type == "ddim":
            noise_scheduler = self.noise_scheduler

        sample = self.velocity_net.sample(
            global_cond,
            timesteps=self.num_inference_steps,
            generator=generator,
            noise_scheduler=noise_scheduler,
        )
        return sample

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]

        # encode text description
        with torch.no_grad():
            language_embedding = self.language_encoder(batch["task"])
        language_cond_feats = self.language_embedding_projection(language_embedding)

        if not self.use_encoder:
            # === FLAT path (original behavior) ===
            global_cond_feats = [language_cond_feats]

            if USE_STATE_PROJ:
                states = einops.rearrange(
                    batch[OBS_STATE], "b s ... -> (b s) ...", b=batch_size, s=n_obs_steps
                )
                states_embedding = self.state_dropout(self.state_proj(states))
                states_feature = einops.rearrange(
                    states_embedding, "(b s) ... -> b s ...", b=batch_size, s=n_obs_steps
                )
                global_cond_feats.append(states_feature)
            else:
                global_cond_feats.append(batch[OBS_STATE].flatten(start_dim=1))

            if self.config.image_features:
                images = einops.rearrange(
                    batch["observation.images"], "b s n ... -> (b s n) ...",
                    b=batch_size, s=n_obs_steps, n=len(self.config.image_features)
                )
                with torch.no_grad():
                    img_cls_tokens = self.pretrained_rgb_encoder(images)
                img_embeddings = self.rgb_embedding_projection(img_cls_tokens)
                img_features = einops.rearrange(
                    img_embeddings, "(b s n) ... -> b s (n ...)",
                    b=batch_size, s=n_obs_steps, n=len(self.config.image_features)
                )
                global_cond_feats.append(img_features.flatten(start_dim=1))

            if self.config.env_state_feature:
                global_cond_feats.append(batch[OBS_ENV_STATE].flatten(start_dim=1))

            return torch.cat(global_cond_feats, dim=-1)  # (B, flat_cond_dim)
        else:
            # === TOKEN path (encoder-based) ===
            global_cond_tokens = []

            # Language: (B, hidden_dim) -> (B, 1, hidden_dim)
            global_cond_tokens.append(language_cond_feats.unsqueeze(1))

            # State: project to hidden_dim, keep temporal tokens
            states = einops.rearrange(
                batch[OBS_STATE], "b s ... -> (b s) ...", b=batch_size, s=n_obs_steps
            )
            states_emb = self.state_dropout(self.state_proj(states))
            states_tokens = einops.rearrange(
                states_emb, "(b s) d -> b s d", b=batch_size, s=n_obs_steps
            )
            global_cond_tokens.append(states_tokens)

            # Images: project, keep as separate tokens per camera per obs step
            if self.config.image_features:
                n_cam = len(self.config.image_features)
                images = einops.rearrange(
                    batch["observation.images"], "b s n ... -> (b s n) ...",
                    b=batch_size, s=n_obs_steps, n=n_cam
                )
                with torch.no_grad():
                    img_cls_tokens = self.pretrained_rgb_encoder(images)
                img_embeddings = self.rgb_embedding_projection(img_cls_tokens)
                img_tokens = einops.rearrange(
                    img_embeddings, "(b s n) d -> b (s n) d",
                    b=batch_size, s=n_obs_steps, n=n_cam
                )
                global_cond_tokens.append(img_tokens)

            if self.config.env_state_feature:
                env_states = einops.rearrange(
                    batch[OBS_ENV_STATE], "b s ... -> (b s) ...", b=batch_size, s=n_obs_steps
                )
                env_emb = self.env_state_proj(env_states)
                env_tokens = einops.rearrange(
                    env_emb, "(b s) d -> b s d", b=batch_size, s=n_obs_steps
                )
                global_cond_tokens.append(env_tokens)

            return torch.cat(global_cond_tokens, dim=1)  # (B, T_total, hidden_dim)

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        trajectory = batch["action"]

        if self.noise_type == "flow_matching":
            # Flow matching: continuous t, velocity prediction
            noise = self.velocity_net.sample_noise(trajectory.shape[0], trajectory.device)
            timesteps = self.noise_distribution.sample(
                (trajectory.shape[0],)
            ).to(trajectory.device)
            noisy_trajectory = (
                (1 - timesteps[:, None, None]) * noise
                + timesteps[:, None, None] * trajectory
            )
            pred = self.velocity_net(
                noisy_actions=noisy_trajectory, time=timesteps, global_cond=global_cond
            )
            target = trajectory - noise

        elif self.noise_type == "ddim":
            # DDIM diffusion: integer timesteps, epsilon/sample prediction
            eps = torch.randn(trajectory.shape, device=trajectory.device)
            timesteps = torch.randint(
                low=0,
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(trajectory.shape[0],),
                device=trajectory.device,
            ).long()
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
            pred = self.velocity_net(
                noisy_actions=noisy_trajectory, time=timesteps, global_cond=global_cond
            )
            if self.config.prediction_type == "epsilon":
                target = eps
            elif self.config.prediction_type == "sample":
                target = trajectory
            else:
                raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
