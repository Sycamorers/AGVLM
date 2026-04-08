from agri_vlm.rewards.classification import normalized_label_reward
from agri_vlm.rewards.clarify_decision import clarify_vs_respond_reward
from agri_vlm.rewards.composite import compute_composite_reward
from agri_vlm.rewards.exact_match import exact_match_reward
from agri_vlm.schemas.reward_schema import RewardInput


def test_exact_match_reward() -> None:
    reward_input = RewardInput(
        prediction="leaf spot",
        task_type="vqa",
        target_text="leaf spot",
        acceptable_answers=["leaf spot"],
    )
    assert exact_match_reward(reward_input) == 1.0


def test_normalized_label_reward() -> None:
    reward_input = RewardInput(
        prediction="Tomato___Early_Blight",
        task_type="classification",
        target_label="tomato early blight",
    )
    assert normalized_label_reward(reward_input) == 1.0


def test_clarify_vs_respond_reward() -> None:
    reward_input = RewardInput(
        prediction="Please upload a clearer close-up image before I answer.",
        task_type="clarify_or_respond",
        expected_decision="clarify",
    )
    assert clarify_vs_respond_reward(reward_input) == 1.0


def test_composite_reward_combines_modules() -> None:
    reward_input = RewardInput(
        prediction="leaf spot",
        task_type="classification",
        target_label="leaf spot",
        acceptable_answers=["leaf spot"],
        weights={"exact_match": 1.0, "normalized_label": 1.0},
    )
    breakdown = compute_composite_reward(
        reward_input,
        reward_modules=["exact_match", "normalized_label"],
        reward_weights={},
    )
    assert breakdown.by_module["exact_match"] == 1.0
    assert breakdown.by_module["normalized_label"] == 1.0
    assert breakdown.total == 2.0
