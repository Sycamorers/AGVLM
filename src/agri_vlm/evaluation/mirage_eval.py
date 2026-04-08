"""MIRAGE evaluation helpers."""

from typing import Any, Dict

from agri_vlm.data.manifest_io import read_manifest
from agri_vlm.evaluation.inference import generate_predictions, oracle_predictions
from agri_vlm.evaluation.metrics import clarify_accuracy, exact_match_rate


def run_mirage_eval(model_config: Any, eval_config: Any) -> Dict[str, Any]:
    rows = read_manifest(eval_config.manifest_path)
    predictions = (
        oracle_predictions(rows)
        if eval_config.prediction_mode == "oracle"
        else generate_predictions(rows, model_config, eval_config.max_new_tokens)
    )

    clarify_refs = [row.target.decision for row in rows if row.target.decision]
    clarify_preds = [prediction for row, prediction in zip(rows, predictions) if row.target.decision]
    answer_refs = [list(row.target.acceptable_answers) or [row.target.answer_text or ""] for row in rows]
    return {
        "num_examples": len(rows),
        "answer_exact_match": exact_match_rate(answer_refs, predictions),
        "clarify_accuracy": clarify_accuracy(clarify_refs, clarify_preds) if clarify_refs else 0.0,
    }
