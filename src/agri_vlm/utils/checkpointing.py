"""Checkpoint discovery helpers."""

from pathlib import Path
from typing import Optional


def _checkpoint_step(path: Path) -> Optional[int]:
    prefix = "checkpoint-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    suffix = path.name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _has_deepspeed_resume_state(path: Path) -> bool:
    latest_path = path / "latest"
    if not latest_path.is_file():
        return False
    tag = latest_path.read_text(encoding="utf-8").strip()
    if not tag:
        return False
    tag_dir = path / tag
    if not tag_dir.is_dir():
        return False
    model_states = list(tag_dir.glob("*_model_states.pt"))
    optim_states = list(tag_dir.glob("*_optim_states.pt"))
    return bool(model_states) and len(model_states) == len(optim_states)


def _has_standard_resume_state(path: Path) -> bool:
    model_files = (
        "pytorch_model.bin",
        "model.safetensors",
        "adapter_model.safetensors",
    )
    return (
        (path / "optimizer.pt").is_file()
        and (path / "scheduler.pt").is_file()
        and any((path / name).is_file() for name in model_files)
    )


def _is_complete_checkpoint(path: Path) -> bool:
    return (
        (path / "trainer_state.json").is_file()
        and (path / "training_args.bin").is_file()
        and (_has_deepspeed_resume_state(path) or _has_standard_resume_state(path))
    )


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = [
        (step, path)
        for path in output_dir.glob("checkpoint-*")
        if (step := _checkpoint_step(path)) is not None and _is_complete_checkpoint(path)
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resolve_resume_checkpoint(output_dir: Path, requested: Optional[str]) -> Optional[Path]:
    if not requested:
        return None
    if requested == "auto":
        return find_latest_checkpoint(output_dir)
    path = Path(requested)
    return path if path.exists() else None
