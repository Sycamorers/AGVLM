from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "vlm_baselines"))

from evaluate_predictions import build_summary_table  # noqa: E402
from utils import write_json  # noqa: E402


def test_summary_table_is_phase_aware(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    write_json(
        metrics_dir / "sft_benchmark_model_val_metrics.json",
        {
            "phase": "sft_benchmark",
            "split": "val",
            "model_name": "model",
            "model_key": "model_key",
            "checkpoint_type": "external_baseline",
            "base_model_name_or_path": "model",
            "num_examples": 2,
            "failure_rate": 0.0,
            "invalid_prediction_rate": 0.0,
            "task_macro_average": 1.0,
            "classification": {"macro_f1": 1.0, "top1_accuracy": 1.0, "weighted_f1": 1.0, "balanced_accuracy": 1.0},
            "short_vqa": {"relaxed_accuracy": 1.0, "exact_match": 1.0, "token_f1": 1.0},
            "clarify_or_respond": {"macro_f1": 0.0},
            "consultation": {"structured_section_compliance": 0.0},
            "benchmark_manifest_path": "splits/sft_val_manifest.jsonl",
            "prediction_path": "predictions/model.jsonl",
        },
    )
    rows = build_summary_table(metrics_dir, metrics_dir / "summary_table.csv")
    assert rows[0]["phase"] == "sft_benchmark"
    assert rows[0]["checkpoint_type"] == "external_baseline"
    assert (metrics_dir / "summary_table.json").exists()
    assert (metrics_dir / "summary_table.md").exists()
    assert "phase" in (metrics_dir / "summary_table.csv").read_text(encoding="utf-8").splitlines()[0]
