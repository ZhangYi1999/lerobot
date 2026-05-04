# DiT Policy Architecture

## Module Structure

```
dit/
├── __init__.py
├── configuration_dit.py   — DiTConfig (dataclass, registered as "dit")
├── modeling_dit.py         — DiTPolicy, all nn.Modules
├── processor_dit.py        — Pre/post-processor pipelines
└── README.md
```

## Model Architecture

```
DiTPolicy (PreTrainedPolicy)
├── observation_encoder (ObservationEncoder)
│   ├── vision_encoder (DINOVisionEncoder | CLIPVisionEncoder)   — shared across cameras
│   │   └── model (AutoModel | CLIPVisionModel)              — frozen by default
│   │   OR
│   ├── vision_encoders (ModuleList)                         — one per camera (use_separate_rgb_encoder_per_camera)
│   ├── image_proj (Linear)                                  — only when image_conditioning="cross_attention"
│   ├── text_encoder (CLIPTextEncoder)                       — only when use_language=True
│   │   ├── text_encoder (CLIPTextModel)                     — frozen
│   │   └── projection (Linear)                              — learnable
│   ├── resize (Resize)                                      — optional (image_resize_shape)
│   └── center_crop / maybe_random_crop (CenterCrop | RandomCrop)  — optional (image_crop_shape)
│
├── noise_predictor (DiffusionTransformer)
│   ├── time_mlp (Sequential)                                — SinusoidalPosEmb → Linear → GELU → Linear → GELU
│   ├── time_mlp_r (Sequential)                              — only when objective="mean_flow"
│   ├── input_proj (Linear)                                  — action_dim → hidden_dim
│   ├── pos_embedding (Parameter)                            — optional (use_positional_encoding)
│   ├── transformer_blocks (ModuleList × num_layers)
│   │   ├── TransformerBlock                                 — when image_conditioning="concat"
│   │   │   ├── attn (RoPEAttention)                         — when use_rope=True
│   │   │   │   ├── qkv_proj (Linear)
│   │   │   │   ├── out_proj (Linear)
│   │   │   │   └── rope (RotaryPositionalEmbedding)
│   │   │   │   OR
│   │   │   ├── multihead_attn (MultiheadAttention)          — when use_rope=False
│   │   │   ├── norm1 (LayerNorm, no affine)
│   │   │   ├── norm2 (LayerNorm, no affine)
│   │   │   ├── mlp (Linear → GELU → Linear)
│   │   │   └── adaLN_modulation (SiLU → Linear)            — outputs 6 × hidden_dim (shift/scale/gate × 2)
│   │   │   OR
│   │   ├── CrossAttentionTransformerBlock                   — when image_conditioning="cross_attention"
│   │   │   ├── attn (RoPEAttention) / multihead_attn       — self-attention (same as above)
│   │   │   ├── cross_attn (MultiheadAttention)              — Q=action tokens, KV=image tokens
│   │   │   ├── norm1 (LayerNorm, no affine)
│   │   │   ├── norm_cross (LayerNorm)
│   │   │   ├── norm2 (LayerNorm, no affine)
│   │   │   ├── mlp (Linear → GELU → Linear)
│   │   │   └── adaLN_modulation (SiLU → Linear)            — 6 × hidden_dim
│   └── output_proj (Linear)                                 — hidden_dim → action_dim
│
└── objective (DiffusionObjective | FlowMatchingObjective | MeanFlowObjective)
    ├── DiffusionObjective
    │   └── noise_scheduler (DDPMScheduler | DDIMScheduler)
    ├── FlowMatchingObjective                                — no extra submodules
    └── MeanFlowObjective
        └── jvp_func (torch.func.jvp | torch.autograd.functional.jvp)
```

## Conditioning Modes

### `image_conditioning="concat"` (default)

All features are flattened to a single 1D vector per sample for AdaLN-Zero modulation:

```
state (B, n_obs_steps, state_dim)  ──flatten──→ (B, n_obs_steps × state_dim)  ─┐
images (B, n_obs_steps, N_cam, C, H, W) ─encode→flatten→ (B, n_obs_steps × N_cam × feat_dim) ─┤── cat ──→ conditioning_vec (B, conditioning_dim)
text (B, text_dim) ──expand──→ (B, n_obs_steps × text_dim)                    ─┘

conditioning_dim = n_obs_steps × (state_dim + N_cam × vision_feat_dim [+ text_dim])
```

DiffusionTransformer receives: `cond_features = cat(timestep_emb, conditioning_vec)`

Each TransformerBlock uses AdaLN-Zero: `adaLN_modulation(cond_features) → shift, scale, gate × 2`

### `image_conditioning="cross_attention"`

Image features are kept as token sequences for cross-attention KV; only state (+text) form the flat conditioning vector:

```
state + text ──flatten──→ conditioning_vec (B, n_obs_steps × (state_dim [+ text_dim]))
images ──encode──→ image_proj ──→ cross_cond (B, n_obs_steps × N_cam, hidden_dim)
```

Each CrossAttentionTransformerBlock: self-attn (AdaLN) → cross-attn (Q=actions, KV=cross_cond) → FFN (AdaLN)

## Objectives

| Objective | Training | Inference | Key Config |
|-----------|----------|-----------|------------|
| `diffusion` | Predict noise/sample from noisy actions | Iterative denoising (DDPM/DDIM) | `noise_scheduler_type`, `num_train_timesteps`, `prediction_type` |
| `flow_matching` | Predict velocity field along linear interpolation path | ODE integration (Euler/RK4) | `sigma_min`, `num_integration_steps`, `timestep_sampling_strategy` |
| `mean_flow` | JVP-based mean velocity targets | One-step or multi-step Euler | `flow_ratio`, `time_distribution`, `use_adaptive_loss`, `do_multi_step_sampling` |

## Processor Pipeline

```
Pre-processor:
  RenameObservations → AddBatchDimension → [Tokenizer (if use_language)] → Device → Normalizer

Post-processor:
  Unnormalizer → Device(cpu)
```

## Key Config Fields

| Field | Default | Description |
|-------|---------|-------------|
| `objective` | `"flow_matching"` | `"diffusion"`, `"flow_matching"`, or `"mean_flow"` |
| `image_conditioning` | `"concat"` | `"concat"` (AdaLN-Zero) or `"cross_attention"` (RDT-style) |
| `vision_encoder_name` | `"facebook/dinov2-base"` | Auto-detects: `clip`→CLIPVisionEncoder, `dino`→DINOVisionEncoder. DINOv3 IDs follow the `facebook/dinov3-vitb16-pretrain-lvd1689m` pattern (gated). |
| `hidden_dim` | `512` | Transformer hidden dimension |
| `num_layers` | `6` | Number of transformer blocks |
| `num_heads` | `8` | Attention heads |
| `use_rope` | `True` | Rotary Position Embedding |
| `n_obs_steps` | `2` | Observation history length |
| `horizon` | `16` | Action prediction horizon |
| `n_action_steps` | `8` | Actions executed per call |
| `use_language` | `True` | Enable CLIP text conditioning |
| `freeze_vision_encoder` | `True` | Freeze vision encoder weights |
| `state_dropout` | `0.1` | State feature dropout |
| `selected_state_keys` | `None` | State keys to use (None = `observation.state` only) |
| `selected_image_keys` | `None` | Image keys to use (None = all) |
