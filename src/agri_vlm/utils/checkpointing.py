"""Checkpoint discovery helpers."""

from pathlib import Path
from typing import Optional


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def resolve_resume_checkpoint(output_dir: Path, requested: Optional[str]) -> Optional[Path]:
    if not requested:
        return None
    if requested == "auto":
        return find_latest_checkpoint(output_dir)
    path = Path(requested)
    return path if path.exists() else None
