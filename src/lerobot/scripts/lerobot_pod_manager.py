#!/usr/bin/env python
"""CLI tool for managing RunPod pods running CLARE training jobs.

Usage:
    # Set your RunPod API key
    export RUNPOD_API_KEY=your_key_here

    # Create a pod
    lerobot-pod-manager create --name "clare-a100" --gpu-type "NVIDIA A100 80GB PCIe"

    # Wait for pod to be ready
    lerobot-pod-manager wait <pod_id>

    # Submit a training job
    lerobot-pod-manager submit <pod_id> --script clare --phase adapter \\
        --policy-path lerobot/smolvla_base --dataset-repo-id user/dataset

    # Check job status
    lerobot-pod-manager logs <pod_id> <job_id>

    # Terminate pod
    lerobot-pod-manager terminate <pod_id>
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx is required. Install with: pip install httpx", file=sys.stderr)
    sys.exit(1)

RUNPOD_API_BASE = "https://api.runpod.io/graphql"
RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
PODS_FILE = Path.home() / ".cache" / "lerobot" / "pods.json"


def _get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("Error: RUNPOD_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    return key


def _load_pods() -> dict:
    if PODS_FILE.exists():
        return json.loads(PODS_FILE.read_text())
    return {}


def _save_pods(pods: dict):
    PODS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PODS_FILE.write_text(json.dumps(pods, indent=2))


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _rest_request(method: str, endpoint: str, data: dict | None = None) -> dict:
    url = f"{RUNPOD_REST_BASE}{endpoint}"
    with httpx.Client(timeout=30) as client:
        if method == "GET":
            resp = client.get(url, headers=_headers())
        elif method == "POST":
            resp = client.post(url, headers=_headers(), json=data)
        elif method == "DELETE":
            resp = client.delete(url, headers=_headers())
        else:
            raise ValueError(f"Unsupported method: {method}")

        if resp.status_code >= 400:
            print(f"API error {resp.status_code}: {resp.text}", file=sys.stderr)
            sys.exit(1)
        return resp.json()


def cmd_create(args):
    """Create a new RunPod pod."""
    payload = {
        "name": args.name,
        "imageName": args.docker_image,
        "gpuCount": args.gpu_count,
        "gpuTypeId": args.gpu_type,
        "volumeInGb": args.volume_gb,
        "containerDiskInGb": args.container_disk_gb,
        "cloudType": args.cloud_type,
        "ports": "8000/http,22/tcp",
    }

    if args.env:
        env_vars = {}
        for item in args.env:
            key, value = item.split("=", 1)
            env_vars[key] = value
        payload["env"] = env_vars

    resp = _rest_request("POST", "/pods", payload)
    pod_id = resp.get("id")

    # Save to local state
    pods = _load_pods()
    pods[pod_id] = {
        "name": args.name,
        "gpu_type": args.gpu_type,
        "docker_image": args.docker_image,
        "status": "STARTING",
        "created_at": datetime.now().isoformat(),
        "public_ip": resp.get("publicIp"),
        "jobs": [],
    }
    _save_pods(pods)

    print(f"Pod created: {pod_id}")
    print(f"  Name: {args.name}")
    print(f"  GPU: {args.gpu_type}")
    print(f"  Status: STARTING")
    print(f"\nWait for it to be ready:")
    print(f"  lerobot-pod-manager wait {pod_id}")


def cmd_status(args):
    """Get pod status."""
    resp = _rest_request("GET", f"/pods/{args.pod_id}")

    status = resp.get("desiredStatus", "UNKNOWN")
    public_ip = resp.get("publicIp")
    port_mappings = resp.get("portMappings")

    # Update local state
    pods = _load_pods()
    if args.pod_id in pods:
        pods[args.pod_id]["status"] = status
        pods[args.pod_id]["public_ip"] = public_ip
        _save_pods(pods)

    print(f"Pod: {args.pod_id}")
    print(f"  Status: {status}")
    print(f"  Public IP: {public_ip or 'N/A'}")

    if port_mappings:
        print(f"  Port mappings:")
        for pm in port_mappings:
            print(f"    {pm.get('privatePort')} -> {pm.get('publicPort')} ({pm.get('type', 'tcp')})")

    # Check API accessibility
    api_url = _get_api_url(resp)
    if api_url:
        try:
            with httpx.Client(timeout=5) as client:
                api_resp = client.get(f"{api_url}/")
                if api_resp.status_code == 200:
                    print(f"  API: accessible at {api_url}")
                    return
        except Exception:
            pass
        print(f"  API: not yet accessible at {api_url}")
    else:
        print(f"  API: no URL available yet")


def cmd_wait(args):
    """Wait for pod to be ready and API to be accessible."""
    print(f"Waiting for pod {args.pod_id} to be ready...")

    for attempt in range(args.timeout // 5):
        try:
            resp = _rest_request("GET", f"/pods/{args.pod_id}")
        except SystemExit:
            print("  Pod not found yet, retrying...")
            time.sleep(5)
            continue

        status = resp.get("desiredStatus", "UNKNOWN")
        api_url = _get_api_url(resp)

        if status == "RUNNING" and api_url:
            try:
                with httpx.Client(timeout=5) as client:
                    api_resp = client.get(f"{api_url}/")
                    if api_resp.status_code == 200:
                        # Update local state
                        pods = _load_pods()
                        if args.pod_id in pods:
                            pods[args.pod_id]["status"] = "RUNNING"
                            pods[args.pod_id]["public_ip"] = resp.get("publicIp")
                            pods[args.pod_id]["api_url"] = api_url
                            _save_pods(pods)

                        print(f"\nPod ready!")
                        print(f"  API URL: {api_url}")
                        print(f"\nSubmit a job:")
                        print(f"  lerobot-pod-manager submit {args.pod_id} --script clare --policy-path ... --dataset-repo-id ...")
                        return
            except Exception:
                pass

        elapsed = (attempt + 1) * 5
        print(f"  [{elapsed}s] status={status}, api={'pending' if not api_url else 'connecting...'}")
        time.sleep(5)

    print(f"\nTimeout after {args.timeout}s. Pod may still be starting.", file=sys.stderr)
    sys.exit(1)


def cmd_submit(args):
    """Submit a training job to a running pod."""
    pods = _load_pods()
    pod_info = pods.get(args.pod_id, {})
    api_url = pod_info.get("api_url")

    if not api_url:
        # Try to discover it
        resp = _rest_request("GET", f"/pods/{args.pod_id}")
        api_url = _get_api_url(resp)
        if not api_url:
            print("Error: Cannot determine pod API URL. Is the pod running?", file=sys.stderr)
            sys.exit(1)

    job_params = {
        "script": args.script,
        "policy_path": args.policy_path,
        "dataset_repo_id": args.dataset_repo_id,
    }

    if args.dataset_root:
        job_params["dataset_root"] = args.dataset_root
    if args.output_dir:
        job_params["output_dir"] = args.output_dir
    if args.batch_size:
        job_params["batch_size"] = args.batch_size
    if args.steps:
        job_params["steps"] = args.steps
    if args.phase:
        job_params["phase"] = args.phase
    if args.adapter_checkpoint_path:
        job_params["adapter_checkpoint_path"] = args.adapter_checkpoint_path
    if args.job_name:
        job_params["job_name"] = args.job_name
    if not args.wandb:
        job_params["wandb_enable"] = False

    # Additional args
    if args.extra:
        additional = {}
        for item in args.extra:
            key, value = item.split("=", 1)
            additional[key] = value
        job_params["additional_args"] = additional

    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{api_url}/jobs", json=job_params)
        if resp.status_code != 200:
            print(f"Error submitting job: {resp.status_code} {resp.text}", file=sys.stderr)
            sys.exit(1)
        job_data = resp.json()

    job_id = job_data["job_id"]

    # Save job to local state
    if args.pod_id in pods:
        pods[args.pod_id].setdefault("jobs", []).append(job_id)
        _save_pods(pods)

    print(f"Job submitted: {job_id}")
    print(f"  Script: {args.script}")
    print(f"  Status: {job_data['status']}")
    print(f"\nCheck progress:")
    print(f"  lerobot-pod-manager logs {args.pod_id} {job_id}")


def cmd_logs(args):
    """Get job logs from a running pod."""
    api_url = _get_pod_api_url(args.pod_id)

    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{api_url}/jobs/{args.job_id}/logs", params={"tail": args.tail})
        if resp.status_code == 404:
            print(f"Job {args.job_id} not found on pod {args.pod_id}", file=sys.stderr)
            sys.exit(1)
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} {resp.text}", file=sys.stderr)
            sys.exit(1)
        data = resp.json()

    # Also get status
    status_resp = httpx.get(f"{api_url}/jobs/{args.job_id}", timeout=10)
    if status_resp.status_code == 200:
        status_data = status_resp.json()
        print(f"Job: {args.job_id}")
        print(f"Status: {status_data.get('status', 'unknown')}")
        progress = status_data.get("progress")
        if progress is not None:
            print(f"Progress: {progress:.1f}%")
        print("---")

    print(data.get("logs", "No logs available"))


def cmd_list(args):
    """List all pods."""
    try:
        resp = _rest_request("GET", "/pods")
    except SystemExit:
        print("Failed to list pods from RunPod API.", file=sys.stderr)
        return

    pod_list = resp if isinstance(resp, list) else resp.get("pods", [])

    if not pod_list:
        print("No pods found.")
        return

    for pod in pod_list:
        pod_id = pod.get("id", "unknown")
        name = pod.get("name", "unnamed")
        status = pod.get("desiredStatus", "UNKNOWN")
        gpu = pod.get("machine", {}).get("gpuDisplayName", "unknown")
        cost = pod.get("costPerHr", 0)
        print(f"  {pod_id}  {name:<20}  {status:<10}  {gpu:<25}  ${cost:.2f}/hr")


def cmd_terminate(args):
    """Terminate a pod."""
    _rest_request("DELETE", f"/pods/{args.pod_id}")

    pods = _load_pods()
    if args.pod_id in pods:
        pods[args.pod_id]["status"] = "TERMINATED"
        pods[args.pod_id]["terminated_at"] = datetime.now().isoformat()
        _save_pods(pods)

    print(f"Pod {args.pod_id} terminated.")


def _get_api_url(pod_response: dict) -> str | None:
    """Extract the API URL from a pod response."""
    public_ip = pod_response.get("publicIp")
    port_mappings = pod_response.get("portMappings")

    if not public_ip or not port_mappings:
        return None

    for pm in port_mappings:
        if pm.get("privatePort") == 8000:
            public_port = pm.get("publicPort")
            if public_port:
                return f"http://{public_ip}:{public_port}"

    return None


def _get_pod_api_url(pod_id: str) -> str:
    """Get API URL for a pod, from cache or by querying RunPod."""
    pods = _load_pods()
    api_url = pods.get(pod_id, {}).get("api_url")
    if api_url:
        return api_url

    resp = _rest_request("GET", f"/pods/{pod_id}")
    api_url = _get_api_url(resp)
    if not api_url:
        print(f"Error: Cannot determine API URL for pod {pod_id}.", file=sys.stderr)
        sys.exit(1)
    return api_url


def main():
    parser = argparse.ArgumentParser(
        prog="lerobot-pod-manager",
        description="Manage RunPod pods for CLARE training",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = subparsers.add_parser("create", help="Create a new pod")
    p_create.add_argument("--name", default="clare-training", help="Pod name")
    p_create.add_argument("--gpu-type", required=True, help='GPU type ID (e.g., "NVIDIA A100 80GB PCIe")')
    p_create.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    p_create.add_argument("--docker-image", default="ghcr.io/lerobot/clare-training:latest", help="Docker image")
    p_create.add_argument("--volume-gb", type=int, default=50, help="Volume size in GB")
    p_create.add_argument("--container-disk-gb", type=int, default=50, help="Container disk size in GB")
    p_create.add_argument("--cloud-type", default="SECURE", choices=["SECURE", "COMMUNITY"], help="Cloud type")
    p_create.add_argument("--env", nargs="*", help="Environment variables (KEY=VALUE)")
    p_create.set_defaults(func=cmd_create)

    # status
    p_status = subparsers.add_parser("status", help="Get pod status")
    p_status.add_argument("pod_id", help="Pod ID")
    p_status.set_defaults(func=cmd_status)

    # wait
    p_wait = subparsers.add_parser("wait", help="Wait for pod to be ready")
    p_wait.add_argument("pod_id", help="Pod ID")
    p_wait.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")
    p_wait.set_defaults(func=cmd_wait)

    # submit
    p_submit = subparsers.add_parser("submit", help="Submit a training job")
    p_submit.add_argument("pod_id", help="Pod ID")
    p_submit.add_argument("--script", required=True, choices=["clare", "er", "packnet", "lora"], help="Training script")
    p_submit.add_argument("--policy-path", required=True, help="Policy path")
    p_submit.add_argument("--dataset-repo-id", required=True, help="Dataset repo ID")
    p_submit.add_argument("--dataset-root", help="Dataset root path in container")
    p_submit.add_argument("--output-dir", help="Output directory in container")
    p_submit.add_argument("--batch-size", type=int, help="Batch size")
    p_submit.add_argument("--steps", type=int, help="Training steps")
    p_submit.add_argument("--phase", choices=["full", "adapter", "discriminator"], help="CLARE phase")
    p_submit.add_argument("--adapter-checkpoint-path", help="Adapter checkpoint path (for discriminator phase)")
    p_submit.add_argument("--job-name", help="Job name")
    p_submit.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable WandB")
    p_submit.add_argument("--extra", nargs="*", help="Extra args (key=value)")
    p_submit.set_defaults(func=cmd_submit)

    # logs
    p_logs = subparsers.add_parser("logs", help="Get job logs")
    p_logs.add_argument("pod_id", help="Pod ID")
    p_logs.add_argument("job_id", help="Job ID")
    p_logs.add_argument("--tail", type=int, default=100, help="Number of lines to show")
    p_logs.set_defaults(func=cmd_logs)

    # list
    p_list = subparsers.add_parser("list", help="List all pods")
    p_list.set_defaults(func=cmd_list)

    # terminate
    p_terminate = subparsers.add_parser("terminate", help="Terminate a pod")
    p_terminate.add_argument("pod_id", help="Pod ID")
    p_terminate.set_defaults(func=cmd_terminate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
