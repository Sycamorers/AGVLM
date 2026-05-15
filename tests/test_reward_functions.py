import json

import pytest
from pydantic import ValidationError

from agri_vlm.rewards.classification import normalized_label_reward
from agri_vlm.rewards.clarify_decision import clarify_vs_respond_reward, infer_decision
from agri_vlm.rewards.composite import build_reward_input, compute_composite_reward, make_trl_reward_function
from agri_vlm.rewards.exact_match import exact_match_reward
from agri_vlm.rewards.hallucination_penalty import hallucination_penalty
from agri_vlm.rewards.management_coverage import management_coverage_reward
from agri_vlm.rewards.parsing import extract_answer_field, extract_decision_field, extract_structured_sections
from agri_vlm.rewards.preference import preference_proxy_reward
from agri_vlm.rewards.structure import structured_format_reward
from agri_vlm.rewards.synonym_match import synonym_match_reward
from agri_vlm.rewards.uncertainty import uncertainty_calibration_reward
from agri_vlm.schemas.reward_schema import RewardInput


def test_reward_input_schema_accepts_accepted_labels_and_forbids_extra() -> None:
    reward_input = RewardInput(
        prediction="Answer: tomato early blight",
        task_type="classification",
        accepted_labels=["tomato early blight"],
    )
    assert reward_input.accepted_labels == ["tomato early blight"]
    with pytest.raises(ValidationError):
        RewardInput(
            prediction="Answer: tomato early blight",
            task_type="classification",
            unexpected_field=True,
        )


def test_build_reward_input_accepts_verifier_accepted_labels() -> None:
    reward_input = build_reward_input(
        prediction="Answer: tomato early blight",
        task_type="classification",
        target_json=json.dumps({"answer_text": "tomato early blight", "canonical_label": "tomato early blight"}),
        verifier_json=json.dumps(
            {
                "mode": "label",
                "accepted_labels": ["tomato early blight"],
                "crop": "tomato",
                "disease": "early blight",
            }
        ),
        reward_meta_json=json.dumps({"weights": {}}),
        metadata_json=json.dumps({"known_facts": ["tomato leaf"], "visual_evidence": ["brown lesions"]}),
    )
    assert reward_input.accepted_labels == ["tomato early blight"]
    assert reward_input.crop == "tomato"
    assert reward_input.known_facts == ["tomato leaf"]


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


def test_normalized_label_uses_accepted_labels() -> None:
    reward_input = RewardInput(
        prediction="Answer: tomato early blight",
        task_type="classification",
        accepted_labels=["tomato early blight"],
    )
    assert normalized_label_reward(reward_input) == 1.0


def test_synonym_match_reward() -> None:
    reward_input = RewardInput(
        prediction="Answer: fire blight",
        task_type="classification",
        synonym_groups={"apple fire blight": ["fire blight"]},
    )
    assert synonym_match_reward(reward_input) == 1.0


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
        prediction=(
            "Diagnosis: tomato leaf spot\n"
            "Evidence: circular brown lesions are visible on leaves\n"
            "Uncertainty: image evidence is limited and needs field confirmation\n"
            "Management: remove infected leaves and improve airflow\n"
            "Follow-up: monitor new growth over the next week"
        ),
        task_type="consultation",
        required_sections=["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
    )
    assert structured_format_reward(reward_input) == 1.0
    weak_input = reward_input.model_copy(update={"prediction": "Diagnosis and Evidence and Management are discussed."})
    assert structured_format_reward(weak_input) == 0.0


def test_structured_format_does_not_reward_headings_only() -> None:
    reward_input = RewardInput(
        prediction="Diagnosis:\nEvidence:\nUncertainty:\nManagement:\nFollow-up:\n",
        task_type="consultation",
        required_sections=["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
    )
    assert structured_format_reward(reward_input) == 0.0


def test_structured_format_penalizes_repeated_empty_headings() -> None:
    prediction = (
        "Diagnosis: tomato leaf spot\nDiagnosis:\n"
        "Evidence: circular brown lesions are visible on leaves\n"
        "Uncertainty: image evidence is limited and needs field confirmation\n"
        "Management: remove infected leaves and improve airflow\n"
        "Follow-up: monitor new growth over the next week"
    )
    reward_input = RewardInput(
        prediction=prediction,
        task_type="consultation",
        required_sections=["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
    )
    assert 0.0 < structured_format_reward(reward_input) < 1.0


def test_management_coverage_requires_unique_meaningful_context() -> None:
    reward_input = RewardInput(
        prediction=(
            "Management: Remove infected leaves promptly and dispose of debris away from the bed. "
            "Improve airflow by pruning crowded foliage and spacing plants."
        ),
        task_type="consultation",
        management_keywords=["remove infected leaves", "improve airflow", "remove infected leaves"],
    )
    assert management_coverage_reward(reward_input) == 0.5


def test_management_coverage_caps_keyword_stuffing() -> None:
    reward_input = RewardInput(
        prediction="Management: prune prune prune prune prune prune prune prune prune.",
        task_type="consultation",
        management_keywords=["prune"],
    )
    assert management_coverage_reward(reward_input) <= 0.25


def test_very_long_repetitive_response_receives_penalty() -> None:
    reward_input = RewardInput(
        prediction=" ".join(["prune"] * 400),
        task_type="consultation",
        management_keywords=["prune"],
    )
    assert management_coverage_reward(reward_input) < 0.0
    assert hallucination_penalty(reward_input) < 0.0


def test_uncertainty_does_not_count_high_confidence_or_confirm() -> None:
    reward_input = RewardInput(
        prediction="High confidence. Confirm this diagnosis.",
        task_type="consultation",
        uncertainty_required=True,
    )
    assert uncertainty_calibration_reward(reward_input) < 0.0


@pytest.mark.parametrize(
    "phrase",
    [
        "I am uncertain because the image is unclear.",
        "There is not enough evidence in the image.",
        "Need more information and a clearer image before diagnosis.",
        "Cannot determine from the image alone.",
    ],
)
def test_real_uncertainty_phrases_work_when_expected(phrase: str) -> None:
    reward_input = RewardInput(
        prediction=phrase,
        task_type="consultation",
        uncertainty_required=True,
    )
    assert uncertainty_calibration_reward(reward_input) == 1.0


def test_uncertainty_phrase_not_rewarded_when_not_expected() -> None:
    reward_input = RewardInput(
        prediction="I am uncertain because the image is unclear.",
        task_type="consultation",
        uncertainty_required=False,
        expected_uncertainty=False,
    )
    assert uncertainty_calibration_reward(reward_input) == 0.0


def test_uncertain_but_definitive_unsupported_diagnosis_gets_no_uncertainty_reward() -> None:
    reward_input = RewardInput(
        prediction="I am uncertain because the image is unclear, but this is definitely tomato late blight.",
        task_type="consultation",
        uncertainty_required=True,
    )
    assert uncertainty_calibration_reward(reward_input) < 0.0


def test_overconfident_unsafe_answer_is_penalized() -> None:
    reward_input = RewardInput(
        prediction="This is definitely the disease. Guaranteed cure. No follow-up needed.",
        task_type="consultation",
        uncertainty_required=True,
    )
    assert hallucination_penalty(reward_input) < 0.0


def test_high_confidence_triggers_hallucination_penalty_when_uncertainty_expected() -> None:
    reward_input = RewardInput(
        prediction="High confidence this is early blight.",
        task_type="consultation",
        uncertainty_required=True,
    )
    assert hallucination_penalty(reward_input) <= -0.5


def test_hallucination_penalty_catches_target_contradiction() -> None:
    reward_input = RewardInput(
        prediction="Answer: potato late blight",
        task_type="classification",
        target_label="tomato early blight",
        accepted_labels=["tomato early blight"],
        crop="tomato",
        disease="early blight",
    )
    assert hallucination_penalty(reward_input) <= -0.5


def test_hallucination_penalty_catches_unsupported_clarify_response() -> None:
    reward_input = RewardInput(
        prediction="Decision: respond\nAnswer: This is definitely rust.",
        task_type="clarify_or_respond",
        expected_decision="clarify",
        uncertainty_required=True,
    )
    assert hallucination_penalty(reward_input) <= -1.0


def test_hallucination_penalty_catches_unsupported_chemical_dosage() -> None:
    reward_input = RewardInput(
        prediction="Apply fungicide at 20 ml per liter and it is safe for pets.",
        task_type="consultation",
    )
    assert hallucination_penalty(reward_input) <= -1.0


def test_hallucination_penalty_catches_fabricated_visual_evidence() -> None:
    reward_input = RewardInput(
        prediction="The image clearly shows white powder on the fruit surface.",
        task_type="consultation",
        visual_evidence=["brown concentric lesions on leaves"],
    )
    assert hallucination_penalty(reward_input) <= -0.5


def test_hallucination_penalty_catches_crop_and_disease_mismatch() -> None:
    reward_input = RewardInput(
        prediction="Diagnosis: potato late blight is visible.",
        task_type="consultation",
        crop="tomato",
        disease="early blight",
    )
    assert hallucination_penalty(reward_input) <= -0.8


def test_preference_proxy_is_optional_and_pairwise_compatible() -> None:
    reward_input = RewardInput(
        prediction="safer chosen answer",
        task_type="consultation",
        chosen_response="safer chosen answer",
        rejected_response="unsafe rejected answer",
        expert_quality_score=0.8,
        safety_score=1.0,
    )
    assert preference_proxy_reward(reward_input) == pytest.approx(0.9)
    assert preference_proxy_reward(reward_input.model_copy(update={"prediction": "unsafe rejected answer"})) == pytest.approx(-0.9)


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
