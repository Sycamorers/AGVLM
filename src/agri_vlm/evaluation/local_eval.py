"""Local holdout evaluation."""

from typing import Any, Dict, List

from agri_vlm.data.manifest_io import read_manifest
from agri_vlm.evaluation.inference import generate_predictions, oracle_predictions
from agri_vlm.evaluation.metrics import accuracy, clarify_decision_metrics, exact_match_rate, macro_f1
from agri_vlm.evaluation.reporting import build_prediction_rows
from agri_vlm.rewards.composite import build_reward_input, compute_composite_reward


def run_local_eval_bundle(model_config: Any, eval_config: Any) -> Dict[str, Any]:
    rows = read_manifest(eval_config.manifest_path)
    if eval_config.max_examples:
        rows = rows[: eval_config.max_examples]
    if eval_config.prediction_mode == "oracle":
        predictions = oracle_predictions(rows)
    elif eval_config.prediction_mode == "model":
        predictions = generate_predictions(
            rows,
            model_config,
            eval_config.max_new_tokens,
            batch_size=eval_config.batch_size,
            checkpoint_path=eval_config.checkpoint_path,
        )
    else:
        predictions = [row.messages[-1].content[-1].text for row in rows]

    label_refs: List[str] = []
    label_preds: List[str] = []
    vqa_refs: List[List[str]] = []
    vqa_preds: List[str] = []
    decision_refs: List[str] = []
    decision_preds: List[str] = []
    reward_totals: List[float] = []

    for row, prediction in zip(rows, predictions):
        if row.target.canonical_label:
            label_refs.append(row.target.canonical_label)
            label_preds.append(prediction)
        if row.target.answer_text:
            refs = list(row.target.acceptable_answers) or [row.target.answer_text]
            vqa_refs.append(refs)
            vqa_preds.append(prediction)
        if row.target.decision:
            decision_refs.append(row.target.decision)
            decision_preds.append(prediction)
        reward_input = build_reward_input(
            prediction=prediction,
            task_type=row.task_type,
            target_json=row.target.model_dump_json(),
            verifier_json=row.verifier.model_dump_json(),
            reward_meta_json=row.reward_meta.model_dump_json(),
        )
        reward_totals.append(
            compute_composite_reward(
                reward_input,
                reward_modules=[
                    "exact_match",
                    "normalized_label",
                    "structured_format",
                    "clarify_vs_respond",
                    "management_coverage",
                    "hallucination_penalty",
                ],
                reward_weights={},
            ).total
        )

    decision_metrics = clarify_decision_metrics(decision_refs, decision_preds) if decision_refs else {}
    metrics = {
        "num_examples": len(rows),
        "label_accuracy": accuracy(label_refs, label_preds),
        "label_macro_f1": macro_f1(tuple(label_refs), tuple(label_preds)) if label_refs else 0.0,
        "answer_exact_match": exact_match_rate(vqa_refs, vqa_preds) if vqa_refs else 0.0,
        "average_reward": sum(reward_totals) / float(len(reward_totals)) if reward_totals else 0.0,
    }
    metrics.update(decision_metrics)
    return {
        "metrics": metrics,
        "predictions": build_prediction_rows(rows, predictions),
    }


def run_local_eval(model_config: Any, eval_config: Any) -> Dict[str, Any]:
    return run_local_eval_bundle(model_config=model_config, eval_config=eval_config)["metrics"]
