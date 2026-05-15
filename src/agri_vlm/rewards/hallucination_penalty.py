"""Hallucination penalties."""

import re
from typing import List

from agri_vlm.rewards.clarify_decision import infer_decision
from agri_vlm.rewards.parsing import (
    any_normalized_phrase,
    extract_answer_field,
    extract_structured_sections,
    max_length_or_repetition_penalty,
    normalize_ag_label,
)
from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import normalize_label, normalize_text


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
    "high confidence",
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
    "chemical control",
    "active ingredient",
]

CHEMICAL_SAFETY_MARKERS = [
    "safe for humans",
    "safe for pets",
    "non toxic",
    "non-toxic",
    "no ppe",
    "no protective gear",
    "safe to eat",
    "harvest immediately after spraying",
]

DOSAGE_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*(?:ml|l|g|kg|oz|lb|lbs|tsp|tbsp|ppm|percent|%)"
    r"(?:\s*/\s*(?:l|liter|gallon|acre|ha|plant))?|\d+\s*(?:x|times)\s+per\s+(?:day|week|month))\b",
    re.IGNORECASE,
)

VISUAL_CLAIM_RE = re.compile(
    r"\b(?:image|photo|picture)\s+(?:clearly\s+)?(?:shows|showing|reveals|contains|has)\s+(?P<claim>[^.!\n;]+)",
    re.IGNORECASE,
)

CROP_TERMS = [
    "apple",
    "bean",
    "bell pepper",
    "cabbage",
    "corn",
    "cotton",
    "cucumber",
    "grape",
    "maize",
    "peach",
    "pepper",
    "potato",
    "rice",
    "soybean",
    "squash",
    "strawberry",
    "tomato",
    "wheat",
]

DISEASE_TERMS = [
    "anthracnose",
    "bacterial spot",
    "black rot",
    "blight",
    "downy mildew",
    "early blight",
    "healthy",
    "late blight",
    "leaf mold",
    "leaf spot",
    "mosaic virus",
    "powdery mildew",
    "rust",
    "scab",
    "septoria",
    "yellow leaf curl",
]


def _support_context(reward_input: RewardInput) -> str:
    return normalize_text(
        " ".join(
            [
                reward_input.target_text or "",
                reward_input.target_label or "",
                " ".join(reward_input.target_labels),
                " ".join(reward_input.accepted_labels),
                " ".join(reward_input.management_keywords),
                " ".join(reward_input.known_facts),
                " ".join(reward_input.allowed_claims),
                " ".join(reward_input.visual_evidence),
            ]
        )
    )


def _label_references(reward_input: RewardInput) -> List[str]:
    labels = list(reward_input.target_labels) + list(reward_input.accepted_labels)
    if reward_input.target_label:
        labels.append(reward_input.target_label)
    return [normalize_ag_label(label) for label in labels if normalize_ag_label(label)]


def _diagnosis_candidate(reward_input: RewardInput) -> str:
    sections = extract_structured_sections(reward_input.prediction)
    diagnosis = sections.get("diagnosis") or sections.get("answer") or ""
    if diagnosis:
        return normalize_ag_label(diagnosis)
    return normalize_ag_label(extract_answer_field(reward_input.prediction))


def _contradicts_target_label(reward_input: RewardInput) -> bool:
    references = _label_references(reward_input)
    if not references:
        return False
    candidate = _diagnosis_candidate(reward_input)
    if not candidate:
        return False
    if any(candidate == label or label in candidate for label in references):
        return False
    has_explicit_answer = "answer:" in normalize_text(reward_input.prediction) or "diagnosis:" in normalize_text(
        reward_input.prediction
    )
    agriculture_terms = set(" ".join(CROP_TERMS + DISEASE_TERMS).split())
    candidate_terms = set(candidate.split())
    return has_explicit_answer and bool(candidate_terms.intersection(agriculture_terms))


def _metadata_mismatch_penalty(reward_input: RewardInput) -> float:
    normalized_prediction = normalize_text(reward_input.prediction)
    penalty = 0.0
    expected_crop = normalize_label(reward_input.crop or "")
    if expected_crop:
        for crop in CROP_TERMS:
            normalized_crop = normalize_label(crop)
            if normalized_crop in expected_crop or expected_crop in normalized_crop:
                continue
            if re.search(r"\b%s\b" % re.escape(normalized_crop), normalized_prediction):
                penalty -= 0.4
                break
    expected_disease = normalize_label(reward_input.disease or reward_input.target_label or "")
    if expected_disease:
        for disease in DISEASE_TERMS:
            normalized_disease = normalize_label(disease)
            if normalized_disease in expected_disease or expected_disease in normalized_disease:
                continue
            if re.search(r"\b%s\b" % re.escape(normalized_disease), normalized_prediction):
                penalty -= 0.4
                break
    return penalty


def _unsupported_visual_evidence_penalty(reward_input: RewardInput) -> float:
    if not reward_input.visual_evidence:
        return 0.0
    allowed = normalize_text(" ".join(reward_input.visual_evidence + reward_input.known_facts + reward_input.allowed_claims))
    penalty = 0.0
    for match in VISUAL_CLAIM_RE.finditer(reward_input.prediction or ""):
        claim = normalize_text(match.group("claim"))
        if not claim:
            continue
        if claim in allowed or any(normalize_text(item) in claim for item in reward_input.visual_evidence):
            continue
        penalty -= 0.5
    return max(-1.0, penalty)


def hallucination_penalty(reward_input: RewardInput) -> float:
    penalty = max_length_or_repetition_penalty(reward_input.prediction, reward_input.task_type)
    if reward_input.forbidden_claims and any_normalized_phrase(reward_input.prediction, reward_input.forbidden_claims):
        penalty -= 1.0
    if any_normalized_phrase(reward_input.prediction, GENERIC_FORBIDDEN_CLAIMS):
        penalty -= 1.0
    if reward_input.unsafe_recommendations and any_normalized_phrase(
        reward_input.prediction, reward_input.unsafe_recommendations
    ):
        penalty -= 1.0
    if any_normalized_phrase(reward_input.prediction, OVERCONFIDENT_MARKERS):
        penalty -= 0.5 if reward_input.uncertainty_required or reward_input.expected_decision == "clarify" else 0.25
    if _contradicts_target_label(reward_input):
        penalty -= 0.5
    if reward_input.expected_decision == "clarify" and infer_decision(reward_input.prediction) == "respond":
        penalty -= 0.75
    target_context = _support_context(reward_input)
    if any_normalized_phrase(reward_input.prediction, CHEMICAL_MARKERS) and not any(
        marker in target_context for marker in ["pesticide", "fungicide", "insecticide", "herbicide", "chemical"]
    ):
        penalty -= 0.5
    if (DOSAGE_RE.search(reward_input.prediction or "") or any_normalized_phrase(reward_input.prediction, CHEMICAL_SAFETY_MARKERS)) and not any(
        marker in target_context for marker in ["dosage", "rate", "label", "ppe", "protective", "safety", "pre harvest"]
    ):
        penalty -= 0.75
    if reward_input.uncertainty_required and any_normalized_phrase(
        reward_input.prediction,
        ["certain diagnosis from image alone", "diagnosis is certain from the image", "image alone proves"],
    ):
        penalty -= 0.5
    penalty += _metadata_mismatch_penalty(reward_input)
    penalty += _unsupported_visual_evidence_penalty(reward_input)
    return penalty
