#!/usr/bin/env python3
"""Evaluate an RL checkpoint or adapter on the configured RL holdout manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from agri_vlm.data.manifest_io import read_manifest
from agri_vlm.evaluation.inference import generate_predictions, oracle_predictions
from agri_vlm.rewards.classification import normalized_label_reward
from agri_vlm.rewards.clarify_decision import clarify_vs_respond_reward
from agri_vlm.rewards.composite import build_reward_input, compute_composite_reward
from agri_vlm.rewards.exact_match import exact_match_reward
from agri_vlm.rewards.hallucination_penalty import hallucination_penalty
from agri_vlm.rewards.management_coverage import management_coverage_reward
from agri_vlm.rewards.structure import structured_format_reward
from agri_vlm.rewards.synonym_match import synonym_match_reward
from agri_vlm.schemas.config_schema import ModelConfigSchema, load_config
from agri_vlm.utils.io import ensure_dir, write_json, write_jsonl
from agri_vlm.utils.text import word_count


REWARD_MODULES = [
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
    parser.add_argument("--model-config", default="configs/model/phi4_reasoning_vision_15b_turin_24g.yaml")
    parser.add_argument("--manifest-path", default="data/manifests/full/rl_local_holdout_eval.jsonl")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--prediction-mode", choices=["model", "oracle"], default="model")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--metrics-output", default="reports/rl_eval_metrics.json")
    parser.add_argument("--samples-output", default="reports/rl_eval_samples.jsonl")
    parser.add_argument("--report-output", default="reports/rl_eval_report.md")
    return parser.parse_args()


def _reward_input(row: Any, prediction: str) -> Any:
    return build_reward_input(
        prediction=prediction,
        task_type=row.task_type,
        target_json=row.target.model_dump_json(),
        verifier_json=row.verifier.model_dump_json(),
        reward_meta_json=row.reward_meta.model_dump_json(),
        metadata_json=json.dumps(row.metadata, ensure_ascii=False),
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / float(len(values)) if values else 0.0


def _group_key(row: Any, group: str) -> str:
    if group == "task_type":
        return row.task_type
    if group == "source_dataset":
        return row.source_dataset
    return "overall"


def _score_rows(rows: List[Any], predictions: List[str], group: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Tuple[Any, str]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        buckets[_group_key(row, group)].append((row, prediction))
    metrics: Dict[str, Dict[str, float]] = {}
    for key, pairs in sorted(buckets.items()):
        label_scores = []
        exact_scores = []
        synonym_scores = []
        structured_scores = []
        clarify_scores = []
        management_scores = []
        hallucination_flags = []
        lengths = []
        composite_scores = []
        for row, prediction in pairs:
            reward_input = _reward_input(row, prediction)
            label_scores.append(normalized_label_reward(reward_input))
            exact_scores.append(exact_match_reward(reward_input))
            synonym_scores.append(synonym_match_reward(reward_input))
            structured_scores.append(structured_format_reward(reward_input))
            clarify_scores.append(clarify_vs_respond_reward(reward_input))
            management_scores.append(management_coverage_reward(reward_input))
            hallucination_flags.append(1.0 if hallucination_penalty(reward_input) < 0 else 0.0)
            lengths.append(float(word_count(prediction)))
            composite_scores.append(
                compute_composite_reward(
                    reward_input=reward_input,
                    reward_modules=REWARD_MODULES,
                    reward_weights={},
                ).total
            )
        metrics[key] = {
            "num_examples": float(len(pairs)),
            "classification_label_accuracy": _mean(label_scores),
            "accepted_answer_accuracy": _mean(exact_scores),
            "synonym_soft_score": _mean(synonym_scores),
            "structured_section_compliance": _mean(structured_scores),
            "clarify_decision_accuracy": _mean(clarify_scores),
            "management_keyword_coverage": _mean(management_scores),
            "hallucination_forbidden_claim_rate": _mean(hallucination_flags),
            "average_completion_length": _mean(lengths),
            "average_composite_reward": _mean(composite_scores),
        }
    return metrics


def _sample_rows(rows: List[Any], predictions: List[str]) -> List[Dict[str, Any]]:
    scored = []
    for row, prediction in zip(rows, predictions):
        reward_input = _reward_input(row, prediction)
        score = compute_composite_reward(
            reward_input=reward_input,
            reward_modules=REWARD_MODULES,
            reward_weights={},
        ).total
        scored.append((score, row, prediction))
    if not scored:
        return []
    ordered = sorted(scored, key=lambda item: item[0])
    picks = [
        ("bad", ordered[0]),
        ("borderline", ordered[len(ordered) // 2]),
        ("good", ordered[-1]),
    ]
    return [
        {
            "bucket": bucket,
            "sample_id": row.sample_id,
            "source_dataset": row.source_dataset,
            "task_type": row.task_type,
            "score": score,
            "prediction": prediction,
            "target": row.target.model_dump(mode="json"),
            "verifier": row.verifier.model_dump(mode="json"),
        }
        for bucket, (score, row, prediction) in picks
    ]


def _write_markdown(metrics: Dict[str, Any], examples: List[Dict[str, Any]], path: Path) -> None:
    lines = [
        "# RL Evaluation Report",
        "",
        "- Manifest: `%s`" % metrics["manifest_path"],
        "- Prediction mode: `%s`" % metrics["prediction_mode"],
        "- Checkpoint path: `%s`" % (metrics.get("checkpoint_path") or ""),
        "- Examples: `%s`" % metrics["overall"]["overall"]["num_examples"],
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics["overall"]["overall"].items():
        lines.append("| %s | %.4f |" % (key, value))
    lines.extend(["", "## Examples", ""])
    for example in examples:
        lines.append("- `%s` `%s` score=`%.4f` sample=`%s`" % (
            example["bucket"],
            example["task_type"],
            example["score"],
            example["sample_id"],
        ))
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if args.prediction_mode == "model" and not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required when --prediction-mode=model")
    model_config = load_config(repo_root / args.model_config, ModelConfigSchema)
    rows = read_manifest(Path(args.manifest_path))
    if args.max_examples:
        rows = rows[: args.max_examples]
    if args.prediction_mode == "oracle":
        predictions = oracle_predictions(rows)
    else:
        predictions = generate_predictions(
            rows,
            model_config=model_config,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint_path,
        )
    metrics = {
        "manifest_path": args.manifest_path,
        "checkpoint_path": args.checkpoint_path,
        "prediction_mode": args.prediction_mode,
        "overall": _score_rows(rows, predictions, "overall"),
        "by_task_type": _score_rows(rows, predictions, "task_type"),
        "by_source_dataset": _score_rows(rows, predictions, "source_dataset"),
    }
    sample_rows = _sample_rows(rows, predictions)
    write_json(Path(args.metrics_output), metrics)
    write_jsonl(Path(args.samples_output), sample_rows)
    _write_markdown(metrics, sample_rows, Path(args.report_output))
    print(json.dumps(metrics["overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
