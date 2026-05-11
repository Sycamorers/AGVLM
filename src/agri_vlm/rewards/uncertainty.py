"""Uncertainty calibration reward."""

from agri_vlm.rewards.parsing import any_normalized_phrase
from agri_vlm.schemas.reward_schema import RewardInput


UNCERTAINTY_MARKERS = [
    "uncertain",
    "not enough evidence",
    "need clearer image",
    "need a clearer image",
    "please upload",
    "limited evidence",
    "cannot confirm",
    "moderate confidence",
    "low confidence",
    "field context",
    "local extension",
]

OVERCONFIDENT_MARKERS = [
    "definitely",
    "certain",
    "certainly",
    "guaranteed",
    "100%",
    "100 percent",
    "no doubt",
]


def uncertainty_calibration_reward(reward_input: RewardInput) -> float:
    if any_normalized_phrase(reward_input.prediction, OVERCONFIDENT_MARKERS):
        return -0.5 if reward_input.uncertainty_required else -0.25
    if not reward_input.uncertainty_required:
        return 0.0
    return 1.0 if any_normalized_phrase(reward_input.prediction, UNCERTAINTY_MARKERS) else 0.0
