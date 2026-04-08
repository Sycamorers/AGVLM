"""Checkpoint discovery helpers."""

from pathlib import Path
from typing import Optional


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    return checkpoints[-1]
