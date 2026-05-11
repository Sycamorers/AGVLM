"""Synonym-based rewards."""

from agri_vlm.rewards.parsing import extract_answer_field, normalize_ag_label
from agri_vlm.schemas.reward_schema import RewardInput


def synonym_match_reward(reward_input: RewardInput) -> float:
    prediction = normalize_ag_label(extract_answer_field(reward_input.prediction))
    best_score = 0.0
    for canonical_label, synonyms in reward_input.synonym_groups.items():
        group = [canonical_label] + list(synonyms)
        if any(normalize_ag_label(term) == prediction for term in group):
            best_score = 1.0
            break
    if best_score == 0.0 and reward_input.target_label:
        best_score = 1.0 if normalize_ag_label(reward_input.target_label) == prediction else 0.0
    return best_score
