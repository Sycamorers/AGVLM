"""Structured output rewards."""

from agri_vlm.rewards.parsing import extract_structured_sections, normalize_ag_label
from agri_vlm.schemas.reward_schema import RewardInput


def structured_format_reward(reward_input: RewardInput) -> float:
    required_sections = reward_input.required_sections
    if not required_sections:
        return 0.0
    parsed = extract_structured_sections(reward_input.prediction)
    present = [
        section
        for section in required_sections
        if normalize_ag_label(section) in parsed and parsed[normalize_ag_label(section)]
    ]
    return len(present) / float(len(required_sections))
