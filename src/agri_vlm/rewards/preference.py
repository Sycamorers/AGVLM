"""Interfaces for future learned or expert preference rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import normalize_text


class PreferenceRewardProvider(Protocol):
    """Backend interface for learned reward models or pairwise judges."""

    name: str

    def score(self, reward_input: RewardInput) -> float:
        """Return a scalar preference score for one generated completion."""


@dataclass(frozen=True)
class NullPreferenceRewardProvider:
    """Default provider used when no learned reward model is configured."""

    name: str = "null_preference_reward"

    def score(self, reward_input: RewardInput) -> float:
        return 0.0


def _quality_score(reward_input: RewardInput) -> float:
    scores = [
        reward_input.preference_score,
        reward_input.expert_quality_score,
        reward_input.agronomic_correctness_score,
        reward_input.management_usefulness_score,
        reward_input.uncertainty_calibration_score,
        reward_input.safety_score,
    ]
    numeric_scores = [float(score) for score in scores if score is not None]
    if not numeric_scores:
        return 1.0
    return max(0.0, min(1.0, sum(numeric_scores) / float(len(numeric_scores))))


def preference_proxy_reward(reward_input: RewardInput) -> float:
    """Optional scaffold for expert pairwise rows.

    This is not a learned reward model. It only scores completions that exactly
    match a provided `chosen_response` or `rejected_response`, which makes it
    useful for data validation and smoke tests while keeping default GRPO
    rule-based.
    """

    if not reward_input.chosen_response or not reward_input.rejected_response:
        return 0.0
    prediction = normalize_text(reward_input.prediction)
    if prediction == normalize_text(reward_input.chosen_response):
        return _quality_score(reward_input)
    if prediction == normalize_text(reward_input.rejected_response):
        return -_quality_score(reward_input)
    return 0.0
