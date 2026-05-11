#!/usr/bin/env python3
"""Evaluate VLM benchmark prediction JSONL files and refresh summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from metrics import evaluate_prediction_records
from utils import BENCHMARK_ROOT, collect_environment_info, git_value, model_slug, read_jsonl, utc_now, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--phase", choices=["sft_benchmark", "rl_benchmark"], default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT / "results" / "metrics"))
    parser.add_argument("--summary-table", default=None)
    parser.add_argument("--refresh-summary-only", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    return parser.parse_args()


def _first(rows: list[dict[str, Any]], key: str, default: Any = None) -> Any:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def evaluate_file(
    predictions_path: Path,
    *,
    model_name: str | None = None,
    model_key: str | None = None,
    phase: str | None = None,
    split: str | None = None,
    output_dir: Path | None = None,
    bootstrap_samples: int = 0,
) -> dict[str, Any]:
    rows = read_jsonl(predictions_path)
    metrics = evaluate_prediction_records(rows, bootstrap_samples=bootstrap_samples)
    inferred_model = model_name or _first(rows, "model_name", "unknown")
    inferred_key = model_key or _first(rows, "model_key", model_slug(str(inferred_model)))
    inferred_phase = phase or _first(rows, "phase", "unknown")
    inferred_split = split or _first(rows, "split", "unknown")
    runtimes = [float(row.get("runtime_seconds") or 0.0) for row in rows if row.get("runtime_seconds") is not None]
    metrics.update(
        {
            "phase": inferred_phase,
            "split": inferred_split,
            "model_name": inferred_model,
            "model_key": inferred_key,
            "model_slug": model_slug(str(inferred_key or inferred_model)),
            "checkpoint_type": _first(rows, "checkpoint_type", ""),
            "base_model_name_or_path": _first(rows, "base_model_name_or_path", ""),
            "adapter_path": _first(rows, "adapter_path", ""),
            "checkpoint_path": _first(rows, "checkpoint_path", ""),
            "benchmark_manifest_path": _first(rows, "benchmark_manifest_path", ""),
            "prediction_path": str(predictions_path),
            "predictions_path": str(predictions_path),
            "avg_runtime_seconds": sum(runtimes) / float(len(runtimes)) if runtimes else 0.0,
            "total_runtime_seconds": sum(runtimes),
            "dtype": _first(rows, "dtype", _first(rows, "inference_dtype", "")),
            "quantization": _first(rows, "quantization", ""),
            "generation_config": _first(rows, "generation_config", {}),
            "git_commit": git_value("rev-parse", "HEAD"),
            "timestamp": utc_now(),
            "command": " ".join(sys.argv),
            "environment": collect_environment_info(None),
        }
    )
    if output_dir is not None:
        output_path = output_dir / ("%s_%s_%s_metrics.json" % (model_slug(str(inferred_phase)), metrics["model_slug"], inferred_split))
        write_json(output_path, metrics)
    return metrics


def _metric_get(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return ""
        value = value.get(part)
    return value if value is not None else ""


def _summary_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": payload.get("phase", ""),
        "split": payload.get("split", ""),
        "model_name": payload.get("model_name", ""),
        "model_key": payload.get("model_key", ""),
        "checkpoint_type": payload.get("checkpoint_type", ""),
        "adapter_path": payload.get("adapter_path", ""),
        "checkpoint_path": payload.get("checkpoint_path", ""),
        "base_model_name_or_path": payload.get("base_model_name_or_path", ""),
        "dtype": payload.get("dtype", ""),
        "quantization": payload.get("quantization", ""),
        "generation_config": json.dumps(payload.get("generation_config", {}), sort_keys=True),
        "num_examples": payload.get("num_examples", ""),
        "failure_rate": payload.get("failure_rate", ""),
        "invalid_prediction_rate": payload.get("invalid_prediction_rate", ""),
        "task_macro_average": payload.get("task_macro_average", ""),
        "classification_top1_accuracy": _metric_get(payload, "classification.top1_accuracy"),
        "classification_macro_f1": _metric_get(payload, "classification.macro_f1"),
        "classification_weighted_f1": _metric_get(payload, "classification.weighted_f1"),
        "classification_balanced_accuracy": _metric_get(payload, "classification.balanced_accuracy"),
        "vqa_exact_match": _metric_get(payload, "short_vqa.exact_match"),
        "vqa_normalized_exact_match": _metric_get(payload, "short_vqa.normalized_exact_match"),
        "vqa_relaxed_accuracy": _metric_get(payload, "short_vqa.relaxed_accuracy"),
        "vqa_token_f1": _metric_get(payload, "short_vqa.token_f1"),
        "vqa_yes_no_accuracy": _metric_get(payload, "short_vqa.yes_no_accuracy"),
        "vqa_numeric_relaxed_accuracy": _metric_get(payload, "short_vqa.numeric_relaxed_accuracy"),
        "clarify_decision_accuracy": _metric_get(payload, "clarify_or_respond.decision_accuracy"),
        "clarify_f1": _metric_get(payload, "clarify_or_respond.clarify_f1"),
        "clarify_macro_f1": _metric_get(payload, "clarify_or_respond.macro_f1"),
        "consultation_structured_section_compliance": _metric_get(payload, "consultation.structured_section_compliance"),
        "consultation_management_keyword_coverage": _metric_get(payload, "consultation.management_keyword_coverage"),
        "consultation_forbidden_claim_rate": _metric_get(payload, "consultation.forbidden_claim_rate"),
        "consultation_overconfidence_rate": _metric_get(payload, "consultation.unsafe_or_overconfident_claim_rate"),
        "benchmark_manifest_path": payload.get("benchmark_manifest_path", ""),
        "prediction_path": payload.get("prediction_path", payload.get("predictions_path", "")),
        "git_commit": payload.get("git_commit", ""),
        "timestamp": payload.get("timestamp", ""),
        "metrics_path": str(path),
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    display_fields = [
        "phase",
        "split",
        "model_key",
        "checkpoint_type",
        "num_examples",
        "task_macro_average",
        "classification_macro_f1",
        "vqa_relaxed_accuracy",
        "clarify_macro_f1",
        "consultation_structured_section_compliance",
    ]
    fields = [field for field in display_fields if field in fieldnames]
    lines = [
        "# VLM Benchmark Summary",
        "",
        "| %s |" % " | ".join(fields),
        "| %s |" % " | ".join("---" for _ in fields),
    ]
    for row in rows:
        lines.append("| %s |" % " | ".join(str(row.get(field, "")) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary_table(metrics_dir: Path, output_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        if path.name.startswith("summary_table"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_summary_row(path, payload))
    fieldnames = [
        "phase",
        "split",
        "model_name",
        "model_key",
        "checkpoint_type",
        "adapter_path",
        "checkpoint_path",
        "base_model_name_or_path",
        "dtype",
        "quantization",
        "generation_config",
        "num_examples",
        "failure_rate",
        "invalid_prediction_rate",
        "task_macro_average",
        "classification_top1_accuracy",
        "classification_macro_f1",
        "classification_weighted_f1",
        "classification_balanced_accuracy",
        "vqa_exact_match",
        "vqa_normalized_exact_match",
        "vqa_relaxed_accuracy",
        "vqa_token_f1",
        "vqa_yes_no_accuracy",
        "vqa_numeric_relaxed_accuracy",
        "clarify_decision_accuracy",
        "clarify_f1",
        "clarify_macro_f1",
        "consultation_structured_section_compliance",
        "consultation_management_keyword_coverage",
        "consultation_forbidden_claim_rate",
        "consultation_overconfidence_rate",
        "benchmark_manifest_path",
        "prediction_path",
        "git_commit",
        "timestamp",
        "metrics_path",
    ]
    write_csv(output_path, rows, fieldnames)
    write_json(output_path.with_suffix(".json"), rows)
    _write_markdown(output_path.with_suffix(".md"), rows, fieldnames)
    return rows


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_table) if args.summary_table else output_dir / "summary_table.csv"
    if args.refresh_summary_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = build_summary_table(output_dir, summary_path)
        print(json.dumps({"summary_table": str(summary_path), "rows": len(rows)}, indent=2, sort_keys=True))
        return 0
    if not args.predictions:
        raise ValueError("--predictions is required unless --refresh-summary-only is used")
    metrics = evaluate_file(
        Path(args.predictions),
        model_name=args.model_name,
        model_key=args.model_key,
        phase=args.phase,
        split=args.split,
        output_dir=output_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    build_summary_table(output_dir, summary_path)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
