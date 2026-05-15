"""Uncertainty calibration reward."""

from agri_vlm.rewards.parsing import any_normalized_phrase
from agri_vlm.schemas.reward_schema import RewardInput


UNCERTAINTY_MARKERS = [
    "uncertain",
    "not enough evidence",
    "unclear image",
    "unclear photo",
    "need more information",
    "need clearer image",
    "need a clearer image",
    "cannot determine from the image alone",
    "cannot tell from the image alone",
    "cannot diagnose from the image alone",
    "please upload",
    "limited evidence",
    "cannot confirm",
    "moderate confidence",
    "low confidence",
    "field context",
    "local extension",
]

UNCERTAINTY_GROUNDING_MARKERS = [
    "image",
    "photo",
    "visible",
    "evidence",
    "symptom",
    "missing",
    "unclear",
    "not enough",
    "limited",
    "need",
    "cannot determine",
    "cannot confirm",
    "field context",
    "lab test",
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
    "high confidence",
]


def uncertainty_calibration_reward(reward_input: RewardInput) -> float:
    if any_normalized_phrase(reward_input.prediction, OVERCONFIDENT_MARKERS):
        return -0.5 if reward_input.uncertainty_required else -0.25
    uncertainty_expected = (
        reward_input.expected_uncertainty
        if reward_input.expected_uncertainty is not None
        else reward_input.uncertainty_required
    )
    if not uncertainty_expected:
        return 0.0
    has_uncertainty = any_normalized_phrase(reward_input.prediction, UNCERTAINTY_MARKERS)
    has_grounding = any_normalized_phrase(reward_input.prediction, UNCERTAINTY_GROUNDING_MARKERS)
    return 1.0 if has_uncertainty and has_grounding else 0.0
