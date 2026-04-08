"""Hallucination penalties."""

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import contains_any


OVERCONFIDENT_MARKERS = ["definitely", "certainly", "guaranteed", "100 percent"]


def hallucination_penalty(reward_input: RewardInput) -> float:
    penalty = 0.0
    if reward_input.forbidden_claims and contains_any(reward_input.prediction, reward_input.forbidden_claims):
        penalty -= 1.0
    if reward_input.uncertainty_required and contains_any(reward_input.prediction, OVERCONFIDENT_MARKERS):
        penalty -= 0.5
    return penalty
