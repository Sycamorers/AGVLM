"""Reward abstractions."""

from typing import Dict

from agri_vlm.schemas.reward_schema import RewardBreakdown, RewardInput


def as_breakdown(total: float, by_module: Dict[str, float], note: str = "") -> RewardBreakdown:
    notes = [note] if note else []
    return RewardBreakdown(total=float(total), by_module=by_module, notes=notes)


def safe_weight(weights: Dict[str, float], name: str, default: float = 1.0) -> float:
    return float(weights.get(name, default))
