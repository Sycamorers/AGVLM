"""Classification rewards."""

from agri_vlm.rewards.parsing import extract_answer_field, normalize_ag_label
from agri_vlm.schemas.reward_schema import RewardInput


def normalized_label_reward(reward_input: RewardInput) -> float:
    labels = list(reward_input.target_labels) + list(reward_input.accepted_labels)
    if reward_input.target_label:
        labels.append(reward_input.target_label)
    normalized_prediction = normalize_ag_label(extract_answer_field(reward_input.prediction))
    normalized_labels = [normalize_ag_label(label) for label in labels if label]
    return 1.0 if normalized_prediction and normalized_prediction in normalized_labels else 0.0
