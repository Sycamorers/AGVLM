"""Synonym-based rewards."""

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import contains_any, normalize_label


def synonym_match_reward(reward_input: RewardInput) -> float:
    prediction = normalize_label(reward_input.prediction)
    best_score = 0.0
    for canonical_label, synonyms in reward_input.synonym_groups.items():
        group = [canonical_label] + list(synonyms)
        if any(normalize_label(term) in prediction for term in group):
            best_score = 1.0
            break
    if best_score == 0.0 and reward_input.target_label:
        best_score = 1.0 if contains_any(prediction, [reward_input.target_label]) else 0.0
    return best_score
