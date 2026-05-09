from agri_vlm.rewards.classification import normalized_label_reward
from agri_vlm.rewards.clarify_decision import clarify_vs_respond_reward, infer_decision
from agri_vlm.rewards.composite import compute_composite_reward, make_trl_reward_function
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


def test_infer_decision_uses_json_clarify() -> None:
    assert infer_decision('{"decision": "clarify", "question": "Which crop is this?"}') == "clarify"


def test_infer_decision_uses_json_respond() -> None:
    assert infer_decision('{"decision": "respond", "answer": "Likely leaf spot."}') == "respond"


def test_infer_decision_plain_clarification_question() -> None:
    assert infer_decision("Could you provide a clearer close-up image of the underside of the leaf?") == "clarify"


def test_infer_decision_complete_answer_with_follow_up_question() -> None:
    prediction = (
        "Diagnosis: likely leaf spot. Evidence: visible circular lesions. "
        "Management: remove infected leaves and monitor spread. Can you share another image if it worsens?"
    )
    assert infer_decision(prediction) == "respond"


def test_infer_decision_uncertain_but_responding_answer() -> None:
    prediction = (
        "I am uncertain, but the visible lesions are consistent with leaf spot. "
        "Management: avoid overhead irrigation and monitor new growth."
    )
    assert infer_decision(prediction) == "respond"


def test_infer_decision_empty_is_not_clarify() -> None:
    assert infer_decision("") == "none"


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


def test_make_trl_reward_function_routes_extra_columns() -> None:
    reward_fn = make_trl_reward_function(
        reward_modules=["exact_match", "normalized_label"],
        reward_weights={"normalized_label": 2.0},
    )
    rewards = reward_fn(
        prompts=["unused"],
        completions=["leaf spot"],
        task_type=["classification"],
        target_json=['{"answer_text": "leaf spot", "canonical_label": "leaf spot"}'],
        verifier_json=['{"mode": "label", "accepted_labels": ["leaf spot"]}'],
        reward_meta_json=['{"weights": {}}'],
        unused_column=["kept"],
    )
    assert rewards == [3.0]
