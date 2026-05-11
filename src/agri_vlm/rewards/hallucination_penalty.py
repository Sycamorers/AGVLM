"""Hallucination penalties."""

from agri_vlm.rewards.parsing import any_normalized_phrase, max_length_or_repetition_penalty
from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import normalize_text


GENERIC_FORBIDDEN_CLAIMS = [
    "guaranteed cure",
    "will cure",
    "no follow-up needed",
    "no follow up needed",
    "do not need follow-up",
    "do not need follow up",
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

CHEMICAL_MARKERS = [
    "apply pesticide",
    "spray pesticide",
    "apply fungicide",
    "spray fungicide",
    "apply insecticide",
    "spray insecticide",
    "apply herbicide",
    "spray herbicide",
    "chemical treatment",
]


def hallucination_penalty(reward_input: RewardInput) -> float:
    penalty = max_length_or_repetition_penalty(reward_input.prediction, reward_input.task_type)
    if reward_input.forbidden_claims and any_normalized_phrase(reward_input.prediction, reward_input.forbidden_claims):
        penalty -= 1.0
    if any_normalized_phrase(reward_input.prediction, GENERIC_FORBIDDEN_CLAIMS):
        penalty -= 1.0
    if reward_input.uncertainty_required and any_normalized_phrase(reward_input.prediction, OVERCONFIDENT_MARKERS):
        penalty -= 0.5
    target_context = normalize_text(" ".join([reward_input.target_text or "", " ".join(reward_input.management_keywords)]))
    if any_normalized_phrase(reward_input.prediction, CHEMICAL_MARKERS) and not any(
        marker in target_context for marker in ["pesticide", "fungicide", "insecticide", "herbicide", "chemical"]
    ):
        penalty -= 0.5
    if reward_input.uncertainty_required and any_normalized_phrase(
        reward_input.prediction,
        ["certain diagnosis from image alone", "diagnosis is certain from the image", "image alone proves"],
    ):
        penalty -= 0.5
    return penalty
