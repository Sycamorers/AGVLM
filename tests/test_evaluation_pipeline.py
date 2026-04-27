from types import SimpleNamespace

from agri_vlm.evaluation.local_eval import run_local_eval_bundle
from agri_vlm.evaluation.metrics import clarify_decision_metrics
from agri_vlm.utils.io import write_jsonl


def _sample(sample_id: str, answer_text: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_dataset": "plantvillage_vqa",
        "task_type": "vqa",
        "split": "holdout",
        "images": ["data/raw/_smoke/leaf.png"],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an agricultural assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data/raw/_smoke/leaf.png"},
                    {"type": "text", "text": "What is visible on this leaf?"},
                ],
            },
        ],
        "target": {
            "answer_text": answer_text,
            "acceptable_answers": [answer_text],
        },
        "metadata": {"crop": "tomato"},
        "verifier": {"mode": "exact_match", "accepted_answers": [answer_text]},
        "reward_meta": {"weights": {"exact_match": 1.0}},
    }


def test_local_eval_bundle_exports_predictions_and_honors_max_examples(tmp_path) -> None:
    manifest_path = tmp_path / "local_holdout.jsonl"
    write_jsonl(
        manifest_path,
        [
            _sample("sample-1", "early blight"),
            _sample("sample-2", "late blight"),
        ],
    )
    eval_config = SimpleNamespace(
        manifest_path=str(manifest_path),
        prediction_mode="oracle",
        max_examples=1,
        batch_size=1,
        max_new_tokens=32,
        checkpoint_path=None,
    )

    result = run_local_eval_bundle(model_config=SimpleNamespace(), eval_config=eval_config)

    assert result["metrics"]["num_examples"] == 1
    assert result["metrics"]["answer_exact_match"] == 1.0
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["sample_id"] == "sample-1"
    assert result["predictions"][0]["prediction"] == "early blight"


def test_clarify_decision_metrics_report_error_modes() -> None:
    metrics = clarify_decision_metrics(
        ["clarify", "clarify", "respond", "respond"],
        [
            "Can you share a clearer close-up image?",
            "This is early blight.",
            "What crop is this?",
            "This is healthy.",
        ],
    )

    assert metrics["clarify_accuracy"] == 0.5
    assert metrics["clarify_precision"] == 0.5
    assert metrics["clarify_recall"] == 0.5
    assert metrics["unnecessary_clarification_rate"] == 0.5
    assert metrics["premature_answer_rate"] == 0.5
