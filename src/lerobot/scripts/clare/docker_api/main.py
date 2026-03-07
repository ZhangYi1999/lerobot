"""FastAPI application for managing CLARE training jobs.

Runs inside the Docker container. Provides REST endpoints to start,
monitor, and cancel training jobs via tmux sessions.
"""

import logging
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lerobot.scripts.clare.docker_api.job_manager import job_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="CLARE Training API", version="1.0.0")


class TrainingJobParams(BaseModel):
    # Script selection
    script: Literal["clare", "er", "packnet", "lora"] = "clare"

    # Required
    policy_path: str
    dataset_repo_id: str

    # Paths
    dataset_root: str = "/data"
    output_dir: str = "/output/run"

    # Training
    batch_size: int = 64
    steps: int = 20000

    # CLARE-specific
    phase: Literal["full", "adapter", "discriminator"] | None = None
    adapter_checkpoint_path: str | None = None

    # General
    wandb_enable: bool = True
    job_name: str = "clare_job"
    additional_args: dict[str, Any] | None = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    start_time: str
    params: dict[str, Any]
    progress: float | None = None
    logs: list[str] | None = None
    error: str | None = None


@app.get("/")
async def root():
    return {"message": "CLARE Training API", "version": "1.0.0"}


@app.post("/jobs", response_model=JobStatus)
async def create_job(params: TrainingJobParams):
    """Start a new training job."""
    job_id = job_manager.start_job(params.model_dump())
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=500, detail="Failed to create job")
    return job_data


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a job."""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_data


@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, tail: int = 100):
    """Get the raw log output of a job."""
    logs = job_manager.get_job_logs(job_id, tail=tail)
    if logs is None:
        raise HTTPException(status_code=404, detail="Job logs not found")
    return {"job_id": job_id, "logs": logs}


@app.get("/jobs", response_model=list[JobStatus])
async def list_jobs():
    """List all jobs."""
    return job_manager.list_jobs()


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Failed to cancel job. Job may not exist or is not running.")
    return {"message": f"Job {job_id} cancelled successfully"}
