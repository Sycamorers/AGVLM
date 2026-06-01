import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "vlm_baselines"))

from build_results_dashboard import load_metric_rows, prediction_examples, render_html  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_dashboard_loads_metrics_and_prediction_examples(tmp_path: Path) -> None:
    prediction_path = tmp_path / "run" / "predictions" / "model-test.jsonl"
    _write_jsonl(
        prediction_path,
        [
            {
                "model_key": "model-a",
                "phase": "sft_benchmark",
                "split": "test",
                "task_type": "classification",
                "sample_id": "class-1",
                "prompt": "Identify the pest.",
                "ground_truth": "aphid",
                "raw_output": "Answer: aphid",
                "parsed_prediction": "aphid",
                "parse_status": "exact",
                "invalid_prediction": False,
            },
            {
                "model_key": "model-a",
                "phase": "sft_benchmark",
                "split": "test",
                "task_type": "vqa",
                "sample_id": "vqa-1",
                "prompt": "Is the leaf yellow?",
                "ground_truth": "Yes",
                "raw_output": "Answer: Yes",
                "parsed_prediction": "Yes",
                "parse_status": "exact",
                "invalid_prediction": False,
            },
        ],
    )
    metrics_path = tmp_path / "run" / "metrics" / "sft-benchmark_model-a_test_metrics.json"
    _write_json(
        metrics_path,
        {
            "phase": "sft_benchmark",
            "split": "test",
            "model_name": "Model A",
            "model_key": "model-a",
            "checkpoint_type": "external_baseline",
            "num_examples": 2,
            "invalid_prediction_rate": 0.0,
            "task_macro_average": 0.5,
            "classification": {"macro_f1": 0.25, "top1_accuracy": 0.25},
            "short_vqa": {"relaxed_accuracy": 0.75, "token_f1": 0.8},
            "clarify_or_respond": {"macro_f1": 0.0},
            "consultation": {"structured_section_compliance": 0.0},
            "prediction_path": str(prediction_path),
        },
    )

    rows = load_metric_rows([tmp_path / "run"], phase="", split="")
    examples = prediction_examples(rows, max_examples_per_task_model=1)
    html = render_html(title="Dashboard", rows=rows, examples=examples, data_path="dashboard_data.json")

    assert rows[0]["model_key"] == "model-a"
    assert rows[0]["classification_macro_f1"] == 0.25
    assert {example["task_type"] for example in examples} == {"classification", "vqa"}
    assert "Answer: aphid" in html
    assert "Task Macro" in html
