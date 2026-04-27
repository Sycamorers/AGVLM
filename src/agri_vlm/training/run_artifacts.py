"""Run directory, logging, and provenance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import subprocess
from typing import Any, Dict, Iterable, List, Optional

from agri_vlm.utils.distributed import DistributedContext
from agri_vlm.utils.io import ensure_dir, write_json, write_yaml


@dataclass(frozen=True)
class RunArtifacts:
    """Resolved paths and logging settings for a training run."""

    run_name: str
    output_dir: Path
    metrics_dir: Path
    metrics_jsonl_path: Path
    legacy_metrics_jsonl_path: Path
    tensorboard_dir: Path
    artifact_dir: Path
    report_to: List[str]
    resolved_config_path: Path
    run_metadata_path: Path
    artifact_manifest_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _config_payload(config: Any) -> Dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    if isinstance(config, dict):
        return dict(config)
    return dict(getattr(config, "__dict__", {}))


def _run_git(repo_root: Path, args: Iterable[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_metadata(repo_root: Path) -> Dict[str, Any]:
    status = _run_git(repo_root, ["status", "--short"])
    return {
        "commit": _run_git(repo_root, ["rev-parse", "HEAD"]),
        "branch": _run_git(repo_root, ["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _gpu_metadata() -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"torch_available": False}
    if not torch.cuda.is_available():
        return {"torch_available": True, "cuda_available": False}
    return {
        "torch_available": True,
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def _selected_environment() -> Dict[str, str]:
    keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_NNODES",
        "SLURM_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "AGRI_VLM_DATA_ROOT",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TMPDIR",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def run_name_from_output_dir(output_dir: Path) -> str:
    """Use the configured run directory as the stable run name when none is provided."""
    return output_dir.name or "agri-vlm-run"


def tensorboard_reporters(report_to: Iterable[str]) -> List[str]:
    """Ensure TensorBoard logging is enabled unless logging is explicitly disabled."""
    reporters = [str(item).strip() for item in report_to if str(item).strip()]
    normalized = {item.lower() for item in reporters}
    if not reporters:
        return ["tensorboard"]
    if "none" in normalized:
        return reporters
    if "tensorboard" not in normalized:
        reporters.append("tensorboard")
    return reporters


def prepare_run_artifacts(
    *,
    stage: str,
    model_config: Any,
    train_config: Any,
    distributed_context: DistributedContext,
    dry_run: bool = False,
) -> RunArtifacts:
    """Create stable run paths and write rank-zero provenance files."""
    repo_root = _repo_root()
    output_dir = ensure_dir(Path(train_config.output_dir))
    run_name = train_config.run_name or run_name_from_output_dir(output_dir)
    metrics_dir = ensure_dir(output_dir / "metrics")
    artifact_dir = ensure_dir(Path(train_config.artifact_dir) if train_config.artifact_dir else output_dir / "artifacts")
    tensorboard_dir = ensure_dir(
        Path(train_config.logging_dir) if train_config.logging_dir else output_dir / "tensorboard"
    )
    artifacts = RunArtifacts(
        run_name=run_name,
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        metrics_jsonl_path=metrics_dir / "train_metrics.jsonl",
        legacy_metrics_jsonl_path=output_dir / "metrics.jsonl",
        tensorboard_dir=tensorboard_dir,
        artifact_dir=artifact_dir,
        report_to=tensorboard_reporters(train_config.report_to),
        resolved_config_path=output_dir / "resolved_config.yaml",
        run_metadata_path=output_dir / "run_metadata.json",
        artifact_manifest_path=output_dir / "artifact_manifest.json",
    )

    if train_config.save_run_metadata and distributed_context.is_main_process:
        model_payload = _config_payload(model_config)
        train_payload = _config_payload(train_config)
        write_yaml(
            artifacts.resolved_config_path,
            {
                "stage": stage,
                "run_name": run_name,
                "model_config": model_payload,
                "train_config": train_payload,
            },
        )
        write_json(
            artifacts.run_metadata_path,
            {
                "stage": stage,
                "run_name": run_name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dry_run": dry_run,
                "hostname": socket.gethostname(),
                "repo_root": str(repo_root),
                "output_dir": str(output_dir),
                "metrics_jsonl_path": str(artifacts.metrics_jsonl_path),
                "legacy_metrics_jsonl_path": str(artifacts.legacy_metrics_jsonl_path),
                "tensorboard_dir": str(artifacts.tensorboard_dir),
                "artifact_dir": str(artifact_dir),
                "report_to": artifacts.report_to,
                "manifest_path": train_payload.get("manifest_path"),
                "eval_manifest_path": train_payload.get("eval_manifest_path"),
                "sft_checkpoint_path": train_payload.get("sft_checkpoint_path"),
                "resume_from_checkpoint": train_payload.get("resume_from_checkpoint"),
                "distributed": distributed_context.as_dict(),
                "git": _git_metadata(repo_root),
                "gpu": _gpu_metadata(),
                "environment": _selected_environment(),
            },
        )
    return artifacts


def write_training_artifact_manifest(artifacts: RunArtifacts, extra: Optional[Dict[str, Any]] = None) -> None:
    """Write a small manifest that points paper scripts to the reusable outputs."""
    checkpoint_dirs = sorted(path.name for path in artifacts.output_dir.glob("checkpoint-*") if path.is_dir())
    payload: Dict[str, Any] = {
        "run_name": artifacts.run_name,
        "output_dir": str(artifacts.output_dir),
        "checkpoints": checkpoint_dirs,
        "metrics_jsonl": str(artifacts.metrics_jsonl_path),
        "legacy_metrics_jsonl": str(artifacts.legacy_metrics_jsonl_path),
        "tensorboard_dir": str(artifacts.tensorboard_dir),
        "resolved_config": str(artifacts.resolved_config_path),
        "run_metadata": str(artifacts.run_metadata_path),
        "artifact_dir": str(artifacts.artifact_dir),
    }
    if extra:
        payload.update(extra)
    write_json(artifacts.artifact_manifest_path, payload)
