"""Management advice coverage reward."""

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import overlap_ratio


def management_coverage_reward(reward_input: RewardInput) -> float:
    if not reward_input.management_keywords:
        return 0.0
    return overlap_ratio(reward_input.management_keywords, reward_input.prediction)
