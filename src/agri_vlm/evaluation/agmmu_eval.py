"""AgMMU-oriented evaluation."""

from typing import Any, Dict

from agri_vlm.data.manifest_io import read_manifest
from agri_vlm.evaluation.inference import generate_predictions, oracle_predictions
from agri_vlm.evaluation.metrics import exact_match_rate


def run_agmmu_eval(model_config: Any, eval_config: Any) -> Dict[str, Any]:
    rows = read_manifest(eval_config.manifest_path)
    predictions = (
        oracle_predictions(rows)
        if eval_config.prediction_mode == "oracle"
        else generate_predictions(rows, model_config, eval_config.max_new_tokens)
    )
    references = [list(row.target.acceptable_answers) or [row.target.answer_text or ""] for row in rows]
    return {
        "num_examples": len(rows),
        "exact_match": exact_match_rate(references, predictions),
    }
