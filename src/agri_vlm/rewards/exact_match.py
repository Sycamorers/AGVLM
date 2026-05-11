"""Exact-match rewards."""

from agri_vlm.rewards.parsing import safe_contains_answer
from agri_vlm.schemas.reward_schema import RewardInput


def exact_match_reward(reward_input: RewardInput) -> float:
    references = list(reward_input.acceptable_answers)
    if reward_input.target_text:
        references.append(reward_input.target_text)
    return 1.0 if safe_contains_answer(reward_input.prediction, references) else 0.0
