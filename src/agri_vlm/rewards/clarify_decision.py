"""Clarify-vs-respond rewards."""

import json
from typing import Optional

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import contains_any, normalize_text, normalize_whitespace


CLARIFY_MARKERS = [
    "please upload",
    "please provide",
    "could you provide",
    "can you share",
    "can you provide",
    "need a clearer image",
    "need more information",
    "need more details",
    "before i can answer",
    "before i can diagnose",
    "what crop",
    "which crop",
    "what variety",
    "where is the plant",
]

SUBSTANTIVE_RESPONSE_MARKERS = [
    "diagnosis",
    "evidence",
    "management",
    "follow-up",
    "likely",
    "appears",
    "consistent with",
    "symptoms indicate",
    "recommend",
    "treat",
    "apply",
    "remove",
    "prune",
    "fungicide",
    "insecticide",
    "disease",
    "pest",
]


def infer_decision(prediction: str) -> str:
    text = normalize_whitespace(prediction)
    if not text:
        return "none"
    json_decision = _extract_json_decision(text)
    if json_decision:
        return json_decision
    normalized = normalize_text(text)
    has_clarify_marker = contains_any(normalized, CLARIFY_MARKERS)
    has_substantive_answer = contains_any(normalized, SUBSTANTIVE_RESPONSE_MARKERS)
    if _is_plain_clarification_question(text, normalized):
        has_clarify_marker = True
    if has_clarify_marker and not has_substantive_answer:
        return "clarify"
    return "respond"


def _extract_json_decision(text: str) -> Optional[str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    decision = payload.get("decision")
    if decision in {"clarify", "respond"}:
        return str(decision)
    return None


def _is_plain_clarification_question(text: str, normalized: str) -> bool:
    if "?" not in text:
        return False
    question_count = text.count("?")
    if question_count > 2:
        return True
    leading_question_markers = [
        "can you",
        "could you",
        "would you",
        "please share",
        "please provide",
        "what crop",
        "which crop",
        "where",
        "when did",
        "how long",
        "do you have",
        "is there",
        "are there",
    ]
    return any(normalized.startswith(marker) for marker in leading_question_markers)


def clarify_vs_respond_reward(reward_input: RewardInput) -> float:
    if not reward_input.expected_decision:
        return 0.0
    predicted_decision = infer_decision(reward_input.prediction)
    return 1.0 if predicted_decision == reward_input.expected_decision else 0.0
