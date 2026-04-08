"""Structured output rewards."""

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import section_headers_present


def structured_format_reward(reward_input: RewardInput) -> float:
    required_sections = reward_input.required_sections
    if not required_sections:
        return 0.0
    present = section_headers_present(reward_input.prediction, required_sections)
    return len(present) / float(len(required_sections))
