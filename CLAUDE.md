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

### 3. RunPod Training Infrastructure (commit `5ad4b56d`)

| File | Purpose |
|------|---------|
| `docker/Dockerfile.clare` | Training image: `runpod/pytorch` base + Python 3.12 + lerobot[libero,peft] + peft_lsy. Code mounted at runtime. |
| `src/lerobot/scripts/clare/docker_api/main.py` | FastAPI inside container: POST/GET/DELETE `/jobs` for training management |
| `src/lerobot/scripts/clare/docker_api/job_manager.py` | tmux-based job lifecycle (start/monitor/cancel), JSON state persistence |
| `src/lerobot/scripts/lerobot_pod_manager.py` | CLI: `create`/`wait`/`submit`/`logs`/`list`/`terminate` RunPod pods via REST API |
| `.claude/skills/runpod/SKILL.md` | Claude Code `/runpod` skill for automated training workflow |

**Entry point added:** `lerobot-pod-manager` in `pyproject.toml`

## Not Yet Done

- [ ] Docker image not yet built or pushed (need to verify `runpod/pytorch` tag availability and build)
- [ ] Import checks on migrated scripts not yet run (need Docker or Linux env for Libero)
- [ ] End-to-end RunPod training test not yet performed
- [ ] `RUNPOD_API_KEY` not configured
- [ ] `.dockerignore` is at workspace root (`XLerobot_workspace/.dockerignore`), not in this repo

## Key Technical Notes

- **Libero env already supports single-task eval** via `task_ids` field in `LiberoEnv` config. No code changes needed.
- **RunPod pytorch images** only go up to Python 3.11. Dockerfile.clare adds Python 3.12 via deadsnakes PPA.
- **Editable install + mount overlay**: Dockerfile does `pip install -e`, runtime mounts `src/` over the same path. The `.pth` file still works → code changes take effect without rebuild.
- **peft_lsy overrides official peft**: installed after lerobot to replace the official `peft` package.
