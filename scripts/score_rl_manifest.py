#!/usr/bin/env python3
"""Score an RL manifest with reward functions without launching training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from agri_vlm.rewards.composite import (
    build_reward_input,
    compute_composite_reward,
    summarize_reward_breakdowns,
)
from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.io import read_jsonl, write_json, write_jsonl


DEFAULT_REWARD_MODULES = [
    "exact_match",
    "normalized_label",
    "synonym_match",
    "structured_format",
    "uncertainty_calibration",
    "clarify_vs_respond",
    "management_coverage",
    "hallucination_penalty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="RL manifest JSONL path.")
    parser.add_argument("--output", required=True, help="Per-sample reward report JSONL path.")
    parser.add_argument("--summary-output", default=None, help="Optional aggregate summary JSON path.")
    parser.add_argument("--predictions-jsonl", default=None, help="Optional JSONL with sample_id and prediction.")
    parser.add_argument(
        "--prediction-field",
        default="__target__",
        help="Dot path to a prediction field in the manifest, or __target__ to score target-like completions.",
    )
    parser.add_argument("--reward-modules", default=",".join(DEFAULT_REWARD_MODULES))
    parser.add_argument("--reward-weights-json", default=None, help="Inline JSON object or path to JSON weights.")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def _load_weights(value: Optional[str]) -> Dict[str, float]:
    if not value:
        return {}
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8") if path.exists() else value)
    if not isinstance(payload, dict):
        raise ValueError("--reward-weights-json must be a JSON object or path to one.")
    return {str(key): float(weight) for key, weight in payload.items()}


def _field_value(row: Dict[str, Any], field_path: str) -> Any:
    value: Any = row
    for part in field_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _target_completion(row: Dict[str, Any]) -> str:
    target = row.get("target") or {}
    verifier = row.get("verifier") or {}
    if target.get("decision"):
        if target["decision"] == "clarify":
            answer = target.get("answer_text") or "Please provide a clearer crop close-up before diagnosis."
            return "Decision: clarify\nClarifying question: %s" % answer
        answer = target.get("answer_text") or (target.get("acceptable_answers") or ["monitor symptoms"])[0]
        return "Decision: respond\nAnswer: %s" % answer
    if verifier.get("required_sections"):
        label = target.get("canonical_label") or target.get("answer_text") or "uncertain agricultural issue"
        keywords = verifier.get("management_keywords") or []
        management = "; ".join(str(item) for item in keywords[:4]) or "monitor symptoms and consult local guidance."
        return "\n".join(
            [
                "Diagnosis: %s" % label,
                "Evidence: Visible symptoms should be compared with crop context before acting.",
                "Uncertainty: The image evidence is limited, so confirm with clearer images or local extension guidance.",
                "Management: %s" % management,
                "Follow-up: Recheck new growth and document whether symptoms spread.",
            ]
        )
    if target.get("answer_text"):
        return "Answer: %s" % target["answer_text"]
    if target.get("canonical_label"):
        return "Answer: %s" % target["canonical_label"]
    if target.get("canonical_labels"):
        return "Answer: %s" % target["canonical_labels"][0]
    if target.get("acceptable_answers"):
        return "Answer: %s" % target["acceptable_answers"][0]
    return ""


def _load_prediction_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    predictions: Dict[str, str] = {}
    for row in read_jsonl(Path(path)):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("Prediction rows require sample_id.")
        if "prediction" not in row:
            raise ValueError("Prediction rows require prediction.")
        predictions[sample_id] = str(row["prediction"])
    return predictions


def _prediction_for_row(
    row: Dict[str, Any],
    *,
    prediction_field: str,
    prediction_map: Dict[str, str],
) -> str:
    sample_id = str(row.get("sample_id") or "")
    if sample_id in prediction_map:
        return prediction_map[sample_id]
    if prediction_field == "__target__":
        return _target_completion(row)
    value = _field_value(row, prediction_field)
    return "" if value is None else str(value)


def score_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    summary_output_path: Path,
    prediction_field: str,
    prediction_map: Dict[str, str],
    reward_modules: List[str],
    reward_weights: Dict[str, float],
    max_samples: int = 0,
) -> Dict[str, Any]:
    report_rows: List[Dict[str, Any]] = []
    breakdowns = []
    for index, row in enumerate(read_jsonl(manifest_path), start=1):
        if max_samples > 0 and len(report_rows) >= max_samples:
            break
        UnifiedSample.model_validate(row)
        prediction = _prediction_for_row(row, prediction_field=prediction_field, prediction_map=prediction_map)
        reward_input = build_reward_input(
            prediction=prediction,
            task_type=str(row.get("task_type") or ""),
            target_json=json.dumps(row.get("target") or {}, ensure_ascii=False),
            verifier_json=json.dumps(row.get("verifier") or {}, ensure_ascii=False),
            reward_meta_json=json.dumps(row.get("reward_meta") or {}, ensure_ascii=False),
            metadata_json=json.dumps(row.get("metadata") or {}, ensure_ascii=False),
            preference_json=json.dumps(row.get("preference") or {}, ensure_ascii=False),
        )
        breakdown = compute_composite_reward(
            reward_input=reward_input,
            reward_modules=reward_modules,
            reward_weights=reward_weights,
        )
        if not math.isfinite(breakdown.total):
            raise FloatingPointError("Non-finite reward for line %s sample_id=%s" % (index, row.get("sample_id")))
        breakdowns.append(breakdown)
        report_rows.append(
            {
                "line": index,
                "sample_id": row.get("sample_id"),
                "task_type": row.get("task_type"),
                "source_dataset": row.get("source_dataset"),
                "prediction": prediction,
                "total": breakdown.total,
                "by_module": breakdown.by_module,
            }
        )
    write_jsonl(output_path, report_rows)
    summary = summarize_reward_breakdowns(breakdowns)
    summary.update(
        {
            "manifest": str(manifest_path),
            "output": str(output_path),
            "reward_modules": reward_modules,
            "reward_weights": reward_weights,
            "scored_rows": len(report_rows),
        }
    )
    write_json(summary_output_path, summary)
    return summary


def main() -> int:
    args = parse_args()
    reward_modules = [item.strip() for item in args.reward_modules.split(",") if item.strip()]
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary = score_manifest(
        manifest_path=Path(args.manifest),
        output_path=output_path,
        summary_output_path=summary_output_path,
        prediction_field=args.prediction_field,
        prediction_map=_load_prediction_map(args.predictions_jsonl),
        reward_modules=reward_modules,
        reward_weights=_load_weights(args.reward_weights_json),
        max_samples=args.max_samples,
    )
    print(
        "scored_rl_manifest=%s rows=%s summary=%s"
        % (args.output, summary["scored_rows"], summary_output_path)
    )
    return 0 if summary["scored_rows"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
