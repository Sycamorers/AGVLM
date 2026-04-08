"""Exact-match rewards."""

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import best_exact_match


def exact_match_reward(reward_input: RewardInput) -> float:
    references = list(reward_input.acceptable_answers)
    if reward_input.target_text:
        references.append(reward_input.target_text)
    return best_exact_match(references, reward_input.prediction)
