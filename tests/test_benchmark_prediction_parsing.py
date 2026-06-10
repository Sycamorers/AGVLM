from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "vlm_baselines"))

from prediction_parsing import (  # noqa: E402
    detect_forbidden_claims,
    detect_overconfidence,
    extract_answer_field,
    extract_decision_field,
    extract_label_from_answer,
    extract_structured_sections,
    normalize_yes_no,
    parse_numeric_answer,
    repetition_stats,
)


def test_answer_field_and_label_parse():
    answer, status = extract_answer_field("Rationale: spots\nAnswer: tomato late blight")
    assert answer == "tomato late blight"
    assert status == "exact"
    parsed = extract_label_from_answer(
        "Answer: tomato late blight",
        ["tomato late blight", "tomato early blight"],
    )
    assert parsed["normalized_prediction"] == "tomato late blight"
    assert parsed["parse_status"] == "exact"
    assert parsed["invalid_prediction"] is False


def test_markdown_and_json_answer_field_parse():
    answer, status = extract_answer_field("- **Answer:** tomato late blight\n- **Evidence:** spots")
    assert answer == "tomato late blight"
    assert status == "exact"

    answer, status = extract_answer_field('{"answer": "tomato late blight", "evidence": "spots"}')
    assert answer == "tomato late blight"
    assert status == "json"

    parsed = extract_label_from_answer(
        "```json\n{\"label\": \"tomato late blight\"}\n```",
        ["tomato late blight", "tomato early blight"],
    )
    assert parsed["normalized_prediction"] == "tomato late blight"
    assert parsed["invalid_prediction"] is False


def test_conflicting_labels_are_ambiguous():
    parsed = extract_label_from_answer(
        "This could be tomato late blight or tomato early blight.",
        ["tomato late blight", "tomato early blight"],
    )
    assert parsed["parse_status"] == "ambiguous"
    assert parsed["invalid_prediction"] is True


def test_unmatched_but_parseable_label_is_not_format_invalid():
    parsed = extract_label_from_answer(
        "Answer: corn leaf blight\nEvidence: brown spots",
        ["tomato late blight", "tomato early blight"],
    )
    assert parsed["parse_status"] == "out_of_label_space"
    assert parsed["invalid_prediction"] is False
    assert parsed["out_of_label_space"] is True


def test_decision_field_and_inferred_status():
    decision, status = extract_decision_field("Decision: clarify\nAnswer: Please send a close-up.")
    assert (decision, status) == ("clarify", "exact")
    decision, status = extract_decision_field("Please provide a closer image of the leaf underside.")
    assert decision == "clarify"
    assert status == "inferred"


def test_structured_sections_require_line_start_headers():
    sections = extract_structured_sections("Diagnosis: blight\nEvidence: lesions\nManagement: prune")
    assert sections["diagnosis"] == "blight"
    assert sections["management"] == "prune"
    assert extract_structured_sections("The diagnosis: blight is likely.") == {}


def test_numeric_yes_no_safety_and_repetition_helpers():
    assert parse_numeric_answer("Answer: about 12.5 leaves") == 12.5
    assert normalize_yes_no("Answer: yes, not no") == ("", "ambiguous")
    assert detect_forbidden_claims("This is a guaranteed cure.")
    assert detect_overconfidence("This is definitely safe and 100% certain.")
    stats = repetition_stats("spray spray spray rotate crops")
    assert stats["token_count"] == 5
    assert stats["repetition_rate"] > 0
