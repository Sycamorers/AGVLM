"""Helpers for single-node distributed execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Dict


@dataclass(frozen=True)
class DistributedContext:
    """Runtime distributed context derived from launcher environment variables."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: str = "cpu"
    backend: str = "none"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        return self.local_rank == 0

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["is_distributed"] = self.is_distributed
        payload["is_main_process"] = self.is_main_process
        payload["is_local_main_process"] = self.is_local_main_process
        return payload


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def get_distributed_context(set_device: bool = False) -> DistributedContext:
    """Read launcher environment variables and return a normalized context."""
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    try:
        import torch
    except Exception:  # pragma: no cover - torch is optional for dry-run tooling
        return DistributedContext(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device="cpu",
            backend="unavailable",
        )

    if torch.cuda.is_available():
        if set_device:
            torch.cuda.set_device(local_rank)
        device = "cuda:%s" % local_rank
        backend = "nccl" if world_size > 1 else "none"
    else:
        device = "cpu"
        backend = "gloo" if world_size > 1 else "none"
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def configure_torch_runtime(tf32: bool = True) -> None:
    """Apply safe CUDA runtime defaults when torch is installed."""
    try:
        import torch
    except Exception:  # pragma: no cover - optional in dry-run mode
        return

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32


def rank_zero_print(message: str) -> None:
    """Print only from global rank zero."""
    if get_distributed_context().is_main_process:
        print(message)
