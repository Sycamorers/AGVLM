"""Uncertainty calibration reward."""

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import contains_any


UNCERTAINTY_MARKERS = [
    "uncertain",
    "not enough evidence",
    "need clearer image",
    "please upload",
    "moderate confidence",
    "high confidence",
    "low confidence",
    "confirm",
]


def uncertainty_calibration_reward(reward_input: RewardInput) -> float:
    if not reward_input.uncertainty_required:
        return 0.0
    return 1.0 if contains_any(reward_input.prediction, UNCERTAINTY_MARKERS) else 0.0
