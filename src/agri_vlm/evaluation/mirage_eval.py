"""MIRAGE evaluation helpers."""

from typing import Any, Dict

from agri_vlm.data.manifest_io import read_manifest
from agri_vlm.evaluation.inference import generate_predictions, oracle_predictions
from agri_vlm.evaluation.metrics import clarify_decision_metrics, exact_match_rate
from agri_vlm.evaluation.reporting import build_prediction_rows


def run_mirage_eval_bundle(model_config: Any, eval_config: Any) -> Dict[str, Any]:
    rows = read_manifest(eval_config.manifest_path)
    if eval_config.max_examples:
        rows = rows[: eval_config.max_examples]
    predictions = (
        oracle_predictions(rows)
        if eval_config.prediction_mode == "oracle"
        else generate_predictions(
            rows,
            model_config,
            eval_config.max_new_tokens,
            batch_size=eval_config.batch_size,
            checkpoint_path=eval_config.checkpoint_path,
        )
    )

    clarify_refs = [row.target.decision for row in rows if row.target.decision]
    clarify_preds = [prediction for row, prediction in zip(rows, predictions) if row.target.decision]
    answer_refs = [list(row.target.acceptable_answers) or [row.target.answer_text or ""] for row in rows]
    decision_metrics = clarify_decision_metrics(clarify_refs, clarify_preds) if clarify_refs else {}
    metrics = {
        "num_examples": len(rows),
        "answer_exact_match": exact_match_rate(answer_refs, predictions),
    }
    metrics.update(decision_metrics)
    return {
        "metrics": metrics,
        "predictions": build_prediction_rows(rows, predictions),
    }


def run_mirage_eval(model_config: Any, eval_config: Any) -> Dict[str, Any]:
    return run_mirage_eval_bundle(model_config=model_config, eval_config=eval_config)["metrics"]
