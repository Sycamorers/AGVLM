"""Classification rewards."""

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import best_exact_match, normalize_label


def normalized_label_reward(reward_input: RewardInput) -> float:
    labels = list(reward_input.target_labels)
    if reward_input.target_label:
        labels.append(reward_input.target_label)
    normalized_prediction = normalize_label(reward_input.prediction)
    normalized_labels = [normalize_label(label) for label in labels if label]
    return best_exact_match(normalized_labels, normalized_prediction)
