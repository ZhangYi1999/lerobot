# LeRobot (CLARE Fork) — Work Log

## Project Overview

This is a fork of HuggingFace LeRobot on the `clare` branch, used for the CLARE (Continual Learning with Adapters) paper. The fork adds custom continual learning training scripts and RunPod training infrastructure.

Related repos in the workspace:
- `peft_lsy/`: Custom PEFT fork with CLARE tuner (func_adapter + discriminator)
- `XLeRobot/`: Robot hardware library (ignore for now)
- `lerobot-training-api/`: Reference RunPod API architecture (read-only reference)

## Completed Work

### 1. Branch Setup (commit `6139b133` base)
- Updated fork's `main` branch from upstream `huggingface/lerobot`
- Rebased `clare` branch onto latest main

### 2. CLARE Training Scripts Migration (commits `ef7fd7a6`, `92bb2cbc`)

Migrated 4 training scripts to the new LeRobot API:

| Script | Path | Description |
|--------|------|-------------|
| `clare.py` | `src/lerobot/scripts/clare/clare.py` | CLARE algorithm with distribution shift detection, layer expansion, adapter + discriminator training. **Decoupled into phases** via `phase` config field (`full`/`adapter`/`discriminator`). |
| `er.py` | `src/lerobot/scripts/clare/er.py` | Experience Replay baseline with replay buffer |
| `packnet.py` | `src/lerobot/scripts/clare/packnet.py` | PackNet weight pruning baseline |
| `lora.py` | `src/lerobot/scripts/clare/lora.py` | SeqLoRA with merge-back-to-backbone |
| `ewc.py` | `src/lerobot/scripts/clare/ewc.py` | EWC baseline: diagonal Fisher + per-task state persistence |

**Key migration changes (all 4 scripts):**
- `GradScaler` → `Accelerator` (HuggingFace accelerate)
- Added `PolicyProcessorPipeline` (preprocessor/postprocessor)
- `eval_policy_with_env_init` → `eval_policy_all` (new eval API)
- `EpisodeAwareSampler` updated to new constructor signature
- `WandBLogger` moved to `lerobot.rl.wandb_utils`, custom modes replaced with key prefixing
- `save_checkpoint` now takes `preprocessor`/`postprocessor` kwargs

**clare.py phase decoupling:**
- `_setup_common()`: shared setup (accelerator, dataset, policy, PEFT, processors)
- `_expand_layers()`: distribution shift detection + layer expansion
- `train_adapter(cfg)`: Phase 1 — expand + train adapters
- `train_discriminator(cfg)`: Phase 2 — load adapter checkpoint + train discriminators
- `train(cfg)`: dispatcher by `cfg.phase`
- New config fields: `phase: Literal["full", "adapter", "discriminator"]`, `adapter_checkpoint_path: Path | None`

### 3. GR00T Training Defaults in CLARE Scripts

All 5 CLARE config classes now override LeRobot generic defaults with GR00T N1.5 values:

| Parameter | Value |
|-----------|-------|
| `seed` | 42 |
| `batch_size` | 32 (16+16 for ER) |
| `steps` | 10_000 |
| `log_freq` / `save_freq` / `eval_freq` | 100 / 10_000 / 10_000 |
| `use_policy_training_preset` | **False** (critical — prevents `validate()` from overwriting optimizer/scheduler) |
| optimizer | AdamW: lr=1e-4, betas=(0.95,0.999), weight_decay=1e-5, grad_clip_norm=1.0 |
| scheduler | cosine, 500 warmup steps (5% of 10_000) |

### 4. RunPod Training Infrastructure (commit `5ad4b56d`)

| File | Purpose |
|------|---------|
| `docker/Dockerfile.clare` | Training image: `runpod/pytorch` base + Python 3.12 + lerobot[all] + peft_lsy. Code mounted at runtime. |
| `src/lerobot/scripts/clare/docker_api/main.py` | FastAPI inside container: POST/GET/DELETE `/jobs` for training management |
| `src/lerobot/scripts/clare/docker_api/job_manager.py` | tmux-based job lifecycle (start/monitor/cancel), JSON state persistence |
| `src/lerobot/scripts/lerobot_pod_manager.py` | CLI: `create`/`wait`/`submit`/`logs`/`list`/`terminate` RunPod pods via REST API |
| `.claude/skills/runpod/SKILL.md` | Claude Code `/runpod` skill for automated training workflow |

**Entry point added:** `lerobot-pod-manager` in `pyproject.toml`

### 5. Docker Image Build & Verification (2026-03-08)

Built and verified `ghcr.io/zhangyi1999/clare-training:latest`:

- **Base image confirmed**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **Fixed**: `apt_pkg` symlink broken in RunPod base image → copied `.so` before `add-apt-repository`
- **Fixed**: `peft_lsy/src/peft/peft_model.py` — `HybridCache` removed in `transformers 5.x`, wrapped import in `try/except`
- **Changed**: lerobot install extras from `[libero,peft]` → `[all]`
- **Verified**: all imports pass (lerobot 0.4.5, peft 0.17.1.dev0, libero, 4 CLARE scripts, docker_api)
- **Verified**: FastAPI starts and responds on port 8000
- **Image pushed**: `ghcr.io/zhangyi1999/clare-training:latest` (confirmed on 2026-03-09; digest: `sha256:e817aee893f64ace5f06e1deb07d26c371d19ce6a399958f14ebd1a15b376169`)
- **Local export**: `C:\Users\Yi\Documents\clare-training.tar.gz` (15GB)

### 6. Continual Learning Experiment Scripts (2026-03-09)

**Dataset verification**: `lerobot/libero_10_subtask` — 500 eps, 10 tasks, 50 eps/task. Confirmed **task i = episodes [i*50 .. i*50+49]** in Docker.

**Bash scripts** created in `lerobot/bash_scripts/`:

| File | Purpose |
|------|---------|
| `common.sh` | Shared config: episode-task mapping, training defaults, WandB, helper functions |
| `verify_dataset.py` | Downloads dataset, verifies episode→task mapping |
| `run_clare.sh` | CLARE: adapter phase → discriminator phase per task |
| `run_er.sh` | ER: current task + replay buffer from all previous tasks |
| `run_packnet.sh` | PackNet: train → prune 75% → post-prune fine-tune |
| `run_lora.sh` | SeqLoRA: LoRA → train → merge into backbone |
| `run_ewc.sh` | EWC: train with Fisher penalty → compute/accumulate Fisher |

**PEFT config files** created in `lerobot/configs/peft/`:

| File | Details |
|------|---------|
| `clare/adapter_config.json` | DiT ff (1536) + VL Self-Attn ff (2048), LoRA rank=32 + Autoencoder discriminator |
| `lora/adapter_config.json` | DiT + VL attention (to_q/k/v/out), LoRA rank=32, alpha=64 |

GR00T N1.5 model dimensions: DiT inner_dim = 32×48 = **1536**, VL self-attn = 32×64 = **2048**.

**Bug fixes**:
- `clare.py`: `phase: Literal[...]` → `str` (draccus CLI parser doesn't support `Literal` decoding)
- `clare.py`: `at_least_expand: Literal[...]` → `str` (same issue)
- `common.sh`: `BASE_MODEL` corrected from non-existent `lerobot/gr00t-1.5b` to `nvidia/GR00T-N1.5-3B`

**RunPod skill** installed: copied `lerobot/.claude/skills/runpod/SKILL.md` → project root `.claude/skills/runpod/SKILL.md`

## Not Yet Done

- [ ] End-to-end training test (blocked locally by CUDA 12.8 vs host driver 550.78 mismatch; needs RunPod or Dockerfile fix)
- [ ] `RUNPOD_API_KEY` not configured
- [ ] `.dockerignore` is at workspace root (`XLerobot_workspace/.dockerignore`), not in this repo

## Key Technical Notes

- **Episode selection already supported natively**: `DatasetConfig.episodes: list[int] | None` — pass episode indices to load only a subset. Implemented via PyArrow predicate pushdown in `datasets/utils.py:load_nested_dataset()`, filters at Parquet level (memory-efficient). CLI: `--dataset.episodes="[0,1,2]"`. Note: `StreamingLeRobotDataset` accepts the param but does **not** filter (known gap); `LeRobotDataset` (non-streaming, used by CLARE scripts) fully supported.
- **Libero env already supports single-task eval** via `task_ids` field in `LiberoEnv` config. No code changes needed.
- **RunPod pytorch images** only go up to Python 3.11. Dockerfile.clare adds Python 3.12 via deadsnakes PPA.
- **Editable install + mount overlay**: Dockerfile does `pip install -e`, runtime mounts `src/` over the same path. The `.pth` file still works → code changes take effect without rebuild.
- **peft_lsy overrides official peft**: installed after lerobot to replace the official `peft` package.
- **Base model is `nvidia/GR00T-N1.5-3B`** (gated, needs HF token). `lerobot/gr00t-1.5b` does NOT exist.
- **Checkpoint structure**: `{output_dir}/checkpoints/last/` (symlink) → `adapter/` (PEFT weights) + `pretrained_model/` (base policy). PackNet adds `pretrained_model/mask.safetensors`. EWC saves Fisher state separately as `ewc_state_task{N}.pt`.
- **Docker CUDA 12.8 incompatible with local RTX 4090** (driver 550.78 supports up to ~CUDA 12.4). Need CUDA 12.4 base image or test on RunPod A100.
