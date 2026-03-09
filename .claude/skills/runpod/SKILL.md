---
name: runpod
description: Manage RunPod GPU training for CLARE. Create pods, submit training jobs (clare/er/packnet/lora), monitor progress, and terminate. Use when user needs remote GPU training, RunPod operations, or end-to-end training management.
argument-hint: [create|submit|monitor|terminate|full-run]
---

# RunPod CLARE Training

## Prerequisites

- `RUNPOD_API_KEY` environment variable must be set
- Docker image built and pushed (see Build section below)
- `httpx` installed locally (`pip install httpx`)

## Full Automated Workflow

When user says "train on RunPod" or similar, execute these steps:

### Step 1: Create Pod

```bash
python -m lerobot.scripts.lerobot_pod_manager create \
  --name "clare-run" \
  --gpu-type "NVIDIA A100 80GB PCIe" \
  --docker-image "ghcr.io/<user>/clare-training:latest" \
  --volume-gb 100
```

Save the returned `pod_id`.

### Step 2: Wait for Pod Ready

```bash
python -m lerobot.scripts.lerobot_pod_manager wait <pod_id>
```

This polls until the pod's FastAPI is accessible. Returns the API URL.

### Step 3: Submit Training Job

```bash
python -m lerobot.scripts.lerobot_pod_manager submit <pod_id> \
  --script clare \
  --phase adapter \
  --policy-path <policy_path> \
  --dataset-repo-id <dataset_repo_id> \
  --output-dir /runpod-volume/outputs/<run_name> \
  --steps 20000
```

Available scripts: `clare`, `er`, `packnet`, `lora`

CLARE phases: `full`, `adapter`, `discriminator`

For discriminator phase, add: `--adapter-checkpoint-path /runpod-volume/outputs/<adapter_run>/checkpoint`

### Step 4: Monitor Progress

```bash
python -m lerobot.scripts.lerobot_pod_manager logs <pod_id> <job_id>
```

Or directly via curl:
```bash
curl http://<api_url>/jobs/<job_id>
curl http://<api_url>/jobs/<job_id>/logs?tail=50
```

### Step 5: Terminate Pod

```bash
python -m lerobot.scripts.lerobot_pod_manager terminate <pod_id>
```

## Other Commands

```bash
# List all pods
python -m lerobot.scripts.lerobot_pod_manager list

# Check pod status
python -m lerobot.scripts.lerobot_pod_manager status <pod_id>
```

## Training Scripts Reference

| Script | Description | Key Args |
|--------|-------------|----------|
| `clare` | CLARE algorithm with distribution shift detection | `--phase full/adapter/discriminator` |
| `er` | Experience Replay baseline with replay buffer | standard args |
| `packnet` | PackNet weight pruning baseline | standard args |
| `lora` | SeqLoRA with merge-back-to-backbone | standard args |

## Build Docker Image

Only needed once (or when dependencies change):

```bash
cd <workspace_root>  # XLerobot_workspace/
docker build -f lerobot/docker/Dockerfile.clare -t clare-training:latest .
docker tag clare-training:latest ghcr.io/<user>/clare-training:latest
docker push ghcr.io/<user>/clare-training:latest
```

## Local Debugging

Same image, mount code for live edits. All persistent data goes to `/runpod-volume`:

```bash
docker run --gpus all -it --rm -p 8000:8000 \
  -v $(pwd)/lerobot/src:/app/lerobot/src \
  -v $(pwd)/peft_lsy/src:/app/peft_lsy/src \
  -v $(pwd)/data:/runpod-volume \
  clare-training:latest /bin/bash
```

On RunPod, `/runpod-volume` is auto-mounted from the network volume — no extra config needed.

Then manually run: `python -m lerobot.scripts.clare.clare --policy.path=... --dataset.repo_id=...`

## Important Notes

- Code is mounted, not baked into image. Update code without rebuilding.
- Only rebuild when Python dependencies change.
- RunPod network volumes persist across pod restarts. Use for datasets and checkpoints.
- Pod hourly cost varies by GPU type. Terminate when done to avoid charges.
- WandB logging is enabled by default. Disable with `--no-wandb`.
