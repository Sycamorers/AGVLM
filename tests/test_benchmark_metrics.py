from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "vlm_baselines"))

from metrics import (  # noqa: E402
    clarify_metrics,
    classification_metrics,
    consultation_metrics,
    evaluate_prediction_records,
    parse_prediction_for_metrics,
    vqa_metrics,
)


def test_classification_scores_and_balanced_accuracy():
    rows = [
        {"ground_truth": "a", "normalized_prediction": "a", "invalid_prediction": False},
        {"ground_truth": "a", "normalized_prediction": "a", "invalid_prediction": False},
        {"ground_truth": "b", "normalized_prediction": "a", "invalid_prediction": False},
    ]
    metrics = classification_metrics(rows)
    assert metrics["top1_accuracy"] == 2 / 3
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["macro_f1"] < metrics["weighted_f1"]


def test_classification_reports_accepted_label_alias_accuracy():
    rows = [
        {
            "ground_truth": "23 corn borer",
            "references": ["23 corn borer", "corn borer", "caterpillar"],
            "normalized_prediction": "caterpillar",
            "invalid_prediction": False,
        }
    ]
    metrics = classification_metrics(rows)
    assert metrics["top1_accuracy"] == 0.0
    assert metrics["accepted_label_accuracy"] == 1.0
    assert metrics["semantic_alias_accuracy"] == 1.0
    assert metrics["out_of_label_space_rate"] == 0.0


def test_classification_out_of_label_space_is_not_invalid_format():
    rows = [
        {
            "ground_truth": "tomato late blight",
            "normalized_prediction": "corn leaf blight",
            "parse_status": "out_of_label_space",
            "invalid_prediction": False,
            "out_of_label_space": True,
        }
    ]
    metrics = classification_metrics(rows)
    assert metrics["invalid_output_rate"] == 0.0
    assert metrics["out_of_label_space_rate"] == 1.0
    assert metrics["top1_accuracy"] == 0.0


def test_classification_parser_marks_ambiguous_invalid():
    parsed = parse_prediction_for_metrics(
        raw_output="Could be tomato late blight or tomato early blight.",
        task_type="classification",
        verifier_mode="label",
        label_space=["tomato late blight", "tomato early blight"],
    )
    assert parsed["parse_status"] == "ambiguous"
    assert parsed["invalid_prediction"] is True


def test_vqa_exact_yes_no_numeric_and_contradiction():
    rows = [
        {"ground_truth": "Tomato", "references": ["Tomato"], "parsed_prediction": "tomato", "normalized_prediction": "tomato"},
        {"ground_truth": "Yes", "references": ["Yes"], "parsed_prediction": "Yes", "normalized_prediction": "yes"},
        {"ground_truth": "10", "references": ["10"], "parsed_prediction": "about 10.2", "normalized_prediction": "about 10.2"},
        {"ground_truth": "No", "references": ["No"], "parsed_prediction": "yes and no", "normalized_prediction": "yes and no"},
    ]
    metrics = vqa_metrics(rows)
    assert metrics["normalized_exact_match"] >= 0.5
    assert metrics["yes_no_accuracy"] == 0.5
    assert metrics["numeric_relaxed_accuracy"] == 1.0
    assert metrics["relaxed_accuracy"] == 0.75


def test_clarify_precision_recall_f1():
    rows = [
        {"ground_truth": "clarify", "normalized_prediction": "clarify"},
        {"ground_truth": "clarify", "normalized_prediction": "respond"},
        {"ground_truth": "respond", "normalized_prediction": "clarify"},
        {"ground_truth": "respond", "normalized_prediction": "respond"},
    ]
    metrics = clarify_metrics(rows)
    assert metrics["decision_accuracy"] == 0.5
    assert metrics["clarify_precision"] == 0.5
    assert metrics["clarify_recall"] == 0.5
    assert metrics["respond_f1"] == 0.5


def test_consultation_metrics_are_structured_not_exact_match():
    rows = [
        {
            "task_type": "consultation",
            "raw_output": (
                "Diagnosis: possible blight\n"
                "Evidence: leaf lesions\n"
                "Uncertainty: confirm with close inspection\n"
                "Management: prune infected leaves and rotate crops\n"
                "Follow-up: send underside image?"
            ),
            "verifier": {
                "required_sections": ["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
                "management_keywords": ["prune", "rotate crops"],
                "forbidden_claims": ["guaranteed cure"],
                "uncertainty_required": True,
            },
        }
    ]
    metrics = consultation_metrics(rows)
    assert metrics["structured_section_compliance"] == 1.0
    assert metrics["management_keyword_coverage"] == 1.0
    assert metrics["forbidden_claim_rate"] == 0.0
    assert metrics["uncertainty_compliance"] == 1.0


def test_evaluate_prediction_records_task_macro():
    rows = [
        {
            "phase": "sft_benchmark",
            "split": "val",
            "task_type": "classification",
            "verifier_mode": "label",
            "ground_truth": "a",
            "normalized_prediction": "a",
            "invalid_prediction": False,
            "source_dataset": "synthetic",
        },
        {
            "phase": "sft_benchmark",
            "split": "val",
            "task_type": "vqa",
            "verifier_mode": "exact_match",
            "ground_truth": "yes",
            "references": ["yes"],
            "parsed_prediction": "yes",
            "normalized_prediction": "yes",
            "invalid_prediction": False,
            "source_dataset": "synthetic",
        },
    ]
    metrics = evaluate_prediction_records(rows)
    assert metrics["task_macro_average"] == 1.0
    assert metrics["per_phase"]["sft_benchmark"]["num_examples"] == 2
    assert "per_source_dataset" in metrics
