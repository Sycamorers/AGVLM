from agri_vlm.rewards.classification import normalized_label_reward
from agri_vlm.rewards.clarify_decision import clarify_vs_respond_reward, infer_decision
from agri_vlm.rewards.composite import compute_composite_reward, make_trl_reward_function
from agri_vlm.rewards.exact_match import exact_match_reward
from agri_vlm.rewards.hallucination_penalty import hallucination_penalty
from agri_vlm.rewards.parsing import extract_answer_field, extract_decision_field, extract_structured_sections
from agri_vlm.rewards.structure import structured_format_reward
from agri_vlm.rewards.uncertainty import uncertainty_calibration_reward
from agri_vlm.schemas.reward_schema import RewardInput


def test_exact_match_reward() -> None:
    reward_input = RewardInput(
        prediction="Answer: leaf spot\nEvidence: circular lesions",
        task_type="vqa",
        target_text="leaf spot",
        acceptable_answers=["leaf spot"],
    )
    assert exact_match_reward(reward_input) == 1.0


def test_normalized_label_reward() -> None:
    reward_input = RewardInput(
        prediction="Answer: Tomato___Early_Blight\nEvidence: spots",
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


def test_field_parsers_use_line_start_fields() -> None:
    text = "Diagnosis: leaf spot\nEvidence: circular lesions\nManagement: remove affected leaves"
    assert extract_answer_field("Answer: leaf spot\nEvidence: spots") == "leaf spot"
    assert extract_decision_field("Decision: clarify\nClarifying question: Which crop?") == "clarify"
    assert extract_structured_sections(text)["diagnosis"] == "leaf spot"


def test_structured_format_requires_true_headers() -> None:
    reward_input = RewardInput(
        prediction="Diagnosis: leaf spot\nEvidence: lesions\nUncertainty: moderate\nManagement: prune\nFollow-up: monitor",
        task_type="consultation",
        required_sections=["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
    )
    assert structured_format_reward(reward_input) == 1.0
    weak_input = reward_input.model_copy(update={"prediction": "Diagnosis and Evidence and Management are discussed."})
    assert structured_format_reward(weak_input) == 0.0


def test_uncertainty_does_not_count_high_confidence_or_confirm() -> None:
    reward_input = RewardInput(
        prediction="High confidence. Confirm this diagnosis.",
        task_type="consultation",
        uncertainty_required=True,
    )
    assert uncertainty_calibration_reward(reward_input) == 0.0


def test_overconfident_unsafe_answer_is_penalized() -> None:
    reward_input = RewardInput(
        prediction="This is definitely the disease. Guaranteed cure. No follow-up needed.",
        task_type="consultation",
        uncertainty_required=True,
    )
    assert hallucination_penalty(reward_input) < 0.0


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
        completions=["Answer: leaf spot"],
        task_type=["classification"],
        target_json=['{"answer_text": "leaf spot", "canonical_label": "leaf spot"}'],
        verifier_json=['{"mode": "label", "accepted_labels": ["leaf spot"]}'],
        reward_meta_json=['{"weights": {}}'],
        unused_column=["kept"],
    )
    assert rewards == [3.0]
