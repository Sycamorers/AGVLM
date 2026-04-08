#!/usr/bin/env python3
"""Verify the local runtime, CUDA stack, and distributed launch environment."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_PACKAGES = ["yaml", "pydantic", "PIL"]
OPTIONAL_PACKAGES = ["torch", "transformers", "datasets", "accelerate", "peft", "trl", "bitsandbytes"]


def try_import(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return "missing (%s)" % exc.__class__.__name__
    return getattr(module, "__version__", "ok")


def run_command(command: List[str]) -> Optional[str]:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def detect_system_cuda() -> Dict[str, Optional[str]]:
    nvcc_output = run_command(["nvcc", "--version"])
    nvidia_smi_output = run_command(["nvidia-smi"])
    return {
        "nvcc_version": nvcc_output.splitlines()[-1] if nvcc_output else None,
        "nvidia_smi_header": nvidia_smi_output.splitlines()[0] if nvidia_smi_output else None,
    }


def detect_gpu_inventory() -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "cuda_available": False,
            "cuda_compiled_version": None,
            "gpu_count": 0,
            "gpu_names": [],
            "bf16_supported": False,
        }

    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_compiled_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": gpu_names,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }


def verify_distributed_runtime() -> Dict[str, Any]:
    try:
        import torch
        import torch.distributed as dist
    except Exception:
        return {"distributed": False, "status": "torch.distributed unavailable"}

    world_size = int((os.environ.get("WORLD_SIZE") or "1"))
    rank = int((os.environ.get("RANK") or "0"))
    local_rank = int((os.environ.get("LOCAL_RANK") or "0"))
    if world_size <= 1:
        return {
            "distributed": False,
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
        }

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    try:
        payload = {
            "distributed": True,
            "backend": dist.get_backend(),
            "world_size": dist.get_world_size(),
            "rank": dist.get_rank(),
            "local_rank": local_rank,
        }
        dist.barrier()
        return payload
    finally:
        if initialized_here:
            dist.destroy_process_group()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print("repo_root=%s" % repo_root)
    print("python=%s" % sys.version.replace("\n", " "))
    print("platform=%s" % platform.platform())

    if sys.version_info[:2] != (3, 11):
        print("ERROR: Python 3.11 is the standardized environment for this repository.")
        return 1

    failures = []
    for package in REQUIRED_PACKAGES:
        version = try_import(package)
        print("required[%s]=%s" % (package, version))
        if version.startswith("missing"):
            failures.append(package)

    for package in OPTIONAL_PACKAGES:
        print("optional[%s]=%s" % (package, try_import(package)))

    print("system_cuda=%s" % json.dumps(detect_system_cuda(), sort_keys=True))
    print("gpu_inventory=%s" % json.dumps(detect_gpu_inventory(), sort_keys=True))
    print("distributed=%s" % json.dumps(verify_distributed_runtime(), sort_keys=True))

    if failures:
        print("ERROR: missing required packages: %s" % ", ".join(failures))
        return 1

    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
