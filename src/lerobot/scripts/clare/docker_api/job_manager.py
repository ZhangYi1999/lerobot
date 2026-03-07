"""Job manager for CLARE training jobs using tmux for persistence.

Adapted from lerobot-training-api/docker-api/job_manager.py.
Manages training jobs via tmux sessions, tracks progress via log files,
and persists job state to JSON.
"""

import json
import logging
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

JOBS_DIR = Path("/app/jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_MAP = {
    "clare": "lerobot.scripts.clare.clare",
    "er": "lerobot.scripts.clare.er",
    "packnet": "lerobot.scripts.clare.packnet",
    "lora": "lerobot.scripts.clare.lora",
}


class JobManager:
    def __init__(self):
        try:
            subprocess.run(["tmux", "-V"], check=True, capture_output=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("tmux is not installed. Job persistence may not work correctly.")
        self.jobs: dict[str, dict[str, Any]] = {}
        self._load_existing_jobs()

    def _load_existing_jobs(self):
        for job_file in JOBS_DIR.glob("*.json"):
            try:
                with open(job_file) as f:
                    job_data = json.load(f)
                    job_id = job_data.get("job_id")
                    if job_id:
                        self.jobs[job_id] = job_data
            except Exception as e:
                logger.error(f"Error loading job file {job_file}: {e}")

    def _save_job_data(self, job_id: str, job_data: dict[str, Any]):
        job_file = JOBS_DIR / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)
        self.jobs[job_id] = job_data

    def _update_job_status(self, job_id: str, status: str, error: str | None = None):
        if job_id in self.jobs:
            job_data = self.jobs[job_id]
            job_data["status"] = status
            if error:
                job_data["error"] = error
            self._save_job_data(job_id, job_data)

    def _parse_progress(self, log_content: str) -> float | None:
        # Match patterns like "step 1000/20000" or "Step 1000/20000"
        matches = re.findall(r"[Ss]tep\s+(\d+)/(\d+)", log_content)
        if matches:
            current_step, total_steps = matches[-1]
            try:
                return (int(current_step) / int(total_steps)) * 100
            except (ValueError, ZeroDivisionError):
                return None
        return None

    def _build_command(self, params: dict[str, Any]) -> str:
        script = params.get("script", "clare")
        module = SCRIPT_MAP.get(script)
        if not module:
            raise ValueError(f"Unknown script: {script}. Available: {list(SCRIPT_MAP.keys())}")

        args = [f"python -m {module}"]

        # Map params to CLI args
        param_to_arg = {
            "policy_path": "policy.path",
            "dataset_repo_id": "dataset.repo_id",
            "dataset_root": "dataset.root",
            "output_dir": "output_dir",
            "batch_size": "batch_size",
            "steps": "steps",
            "phase": "phase",
            "adapter_checkpoint_path": "adapter_checkpoint_path",
            "job_name": "job_name",
        }

        for param_key, arg_key in param_to_arg.items():
            value = params.get(param_key)
            if value is not None:
                args.append(f"--{arg_key}={value}")

        # WandB
        wandb_enable = params.get("wandb_enable", True)
        args.append(f"--wandb.enable={'true' if wandb_enable else 'false'}")

        # Additional args
        if params.get("additional_args"):
            for key, value in params["additional_args"].items():
                args.append(f"--{key}={value}")

        return " ".join(args)

    def start_job(self, params: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        session_name = f"clare_job_{job_id[:8]}"
        log_file = JOBS_DIR / f"{job_id}.log"

        job_data = {
            "job_id": job_id,
            "status": "starting",
            "start_time": datetime.now().isoformat(),
            "params": params,
            "progress": 0,
            "logs": [],
        }
        self._save_job_data(job_id, job_data)

        try:
            cmd_str = self._build_command(params)
        except ValueError as e:
            self._update_job_status(job_id, "error", str(e))
            return job_id

        try:
            create_session_cmd = [
                "tmux", "new-session",
                "-d", "-s", session_name,
                f"cd /app && {cmd_str} 2>&1 | tee {log_file} ; echo 'Job completed with exit code '$?' >> {log_file}",
            ]
            subprocess.run(create_session_cmd, check=True)
            self._update_job_status(job_id, "running")
            self._start_monitoring_thread(job_id, session_name, log_file)
            return job_id
        except Exception as e:
            logger.error(f"Failed to start job: {e}")
            self._update_job_status(job_id, "error", str(e))
            return job_id

    def _start_monitoring_thread(self, job_id: str, session_name: str, log_file: Path):
        def monitor_job():
            try:
                while True:
                    session_exists = subprocess.run(
                        ["tmux", "has-session", "-t", session_name],
                        capture_output=True, check=False,
                    ).returncode == 0

                    if log_file.exists():
                        log_content = log_file.read_text()
                        progress = self._parse_progress(log_content)

                        if job_id in self.jobs:
                            job_data = self.jobs[job_id]
                            # Keep last 200 lines for API response
                            job_data["logs"] = log_content.splitlines()[-200:]
                            if progress is not None:
                                job_data["progress"] = progress

                            if "Job completed with exit code" in log_content:
                                exit_match = re.search(r"Job completed with exit code (\d+)", log_content)
                                if exit_match:
                                    exit_code = int(exit_match.group(1))
                                    job_data["status"] = "completed" if exit_code == 0 else "failed"
                                    if exit_code != 0:
                                        job_data["error"] = f"Process exited with code {exit_code}"
                                else:
                                    job_data["status"] = "completed"
                            self._save_job_data(job_id, job_data)

                    if not session_exists and job_id in self.jobs:
                        job_data = self.jobs[job_id]
                        if job_data["status"] not in ("completed", "failed", "cancelled"):
                            job_data["status"] = "failed"
                            job_data["error"] = "tmux session ended unexpectedly"
                            self._save_job_data(job_id, job_data)
                            break

                    if job_id in self.jobs and self.jobs[job_id]["status"] in ("completed", "failed", "cancelled"):
                        break

                    time.sleep(10)
            except Exception as e:
                logger.error(f"Error monitoring job {job_id}: {e}")
                if job_id in self.jobs:
                    self._update_job_status(job_id, "error", str(e))

        thread = threading.Thread(target=monitor_job, daemon=True)
        thread.start()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        if job_id in self.jobs:
            return self.jobs[job_id]
        job_file = JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            try:
                with open(job_file) as f:
                    job_data = json.load(f)
                    self.jobs[job_id] = job_data
                    return job_data
            except Exception as e:
                logger.error(f"Error reading job file {job_file}: {e}")
        return None

    def list_jobs(self) -> list[dict[str, Any]]:
        self._load_existing_jobs()
        return list(self.jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        job_data = self.get_job(job_id)
        if not job_data or job_data["status"] not in ("running", "starting"):
            return False

        session_name = f"clare_job_{job_id[:8]}"
        try:
            session_exists = subprocess.run(
                ["tmux", "has-session", "-t", session_name],
                capture_output=True, check=False,
            ).returncode == 0
            if session_exists:
                subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            self._update_job_status(job_id, "cancelled", "Job cancelled by user")
            return True
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False

    def get_job_logs(self, job_id: str, tail: int = 100) -> str | None:
        log_file = JOBS_DIR / f"{job_id}.log"
        if not log_file.exists():
            return None
        lines = log_file.read_text().splitlines()
        return "\n".join(lines[-tail:])


job_manager = JobManager()
