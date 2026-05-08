"""Shared utilities for the isolated VLM baseline benchmark."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
from typing import Any, Iterable, Mapping


BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_ROOT.parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_inference_environment() -> None:
    """Set benchmark-safe defaults without touching training artifacts."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(REPO_ROOT / ".cache" / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(REPO_ROOT / ".cache" / "huggingface" / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()
    return slug or "model"


def normalize_text(text: str | None) -> str:
    value = (text or "").lower()
    value = re.sub(r"[^a-z0-9\s:/_-]+", " ", value)
    value = value.replace("/", " ").replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def stable_hash(value: str, seed: int = 42) -> str:
    return hashlib.sha256(("%s::%s" % (seed, value)).encode("utf-8")).hexdigest()


def stable_fraction(value: str, seed: int = 42) -> float:
    digest = stable_hash(value, seed)
    return int(digest[:12], 16) / float(16**12)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSONL at %s:%s: %s" % (path, line_number, exc)) from exc
    return rows


def stream_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSONL at %s:%s: %s" % (path, line_number, exc)) from exc


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload or {}


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def resolve_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def package_version(package_name: str) -> str | None:
    try:
        module = importlib.import_module(package_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "installed"))


def collect_environment_info(device: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "python": sys.version.replace("\n", " "),
        "cwd": str(REPO_ROOT),
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "dirty": bool(git_value("status", "--short")),
        },
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "bitsandbytes": package_version("bitsandbytes"),
            "PIL": package_version("PIL"),
            "qwen_vl_utils": package_version("qwen_vl_utils"),
        },
        "cuda": {
            "requested_device": device,
            "available": False,
            "device_count": 0,
            "devices": [],
        },
    }
    try:
        import torch

        info["cuda"]["available"] = bool(torch.cuda.is_available())
        info["cuda"]["device_count"] = int(torch.cuda.device_count())
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "capability": "%s.%s" % (props.major, props.minor),
                }
            )
        info["cuda"]["devices"] = devices
    except Exception as exc:
        info["cuda"]["error"] = "%s: %s" % (type(exc).__name__, exc)
    return info


def maybe_cuda_memory(device: str | None = None) -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        index = 0
        if device and ":" in device:
            index = int(device.split(":", 1)[1])
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        return {
            "available": True,
            "device": index,
            "free_gb": round(free_bytes / (1024**3), 2),
            "total_gb": round(total_bytes / (1024**3), 2),
            "allocated_gb": round(torch.cuda.memory_allocated(index) / (1024**3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved(index) / (1024**3), 2),
        }
    except Exception as exc:
        return {"available": False, "error": "%s: %s" % (type(exc).__name__, exc)}
