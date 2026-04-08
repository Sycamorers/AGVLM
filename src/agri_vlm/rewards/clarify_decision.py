"""Clarify-vs-respond rewards."""

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import contains_any, normalize_text


CLARIFY_MARKERS = [
    "please upload",
    "can you share",
    "need a clearer image",
    "before i can answer",
    "what crop",
    "?",
]


def infer_decision(prediction: str) -> str:
    normalized = normalize_text(prediction)
    if contains_any(normalized, CLARIFY_MARKERS):
        return "clarify"
    return "respond"


def clarify_vs_respond_reward(reward_input: RewardInput) -> float:
    if not reward_input.expected_decision:
        return 0.0
    predicted_decision = infer_decision(reward_input.prediction)
    return 1.0 if predicted_decision == reward_input.expected_decision else 0.0
