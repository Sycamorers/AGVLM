#!/usr/bin/env python3
"""Evaluate VLM baseline prediction JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metrics import evaluate_prediction_records
from utils import BENCHMARK_ROOT, model_slug, read_jsonl, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT / "results" / "metrics"))
    parser.add_argument("--summary-table", default=None)
    parser.add_argument("--refresh-summary-only", action="store_true")
    return parser.parse_args()


def evaluate_file(
    predictions_path: Path,
    *,
    model_name: str | None = None,
    split: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(predictions_path)
    metrics = evaluate_prediction_records(rows)
    inferred_model = model_name or (rows[0].get("model_name") if rows else "unknown")
    inferred_split = split or (rows[0].get("split") if rows else "unknown")
    runtimes = [float(row.get("runtime_seconds") or 0.0) for row in rows if row.get("runtime_seconds") is not None]
    metrics.update(
        {
            "model_name": inferred_model,
            "model_slug": model_slug(str(inferred_model)),
            "split": inferred_split,
            "predictions_path": str(predictions_path),
            "avg_runtime_seconds": sum(runtimes) / float(len(runtimes)) if runtimes else 0.0,
            "total_runtime_seconds": sum(runtimes),
            "dtype": rows[0].get("inference_dtype") if rows else None,
            "quantization": rows[0].get("quantization") if rows else None,
            "generation_config": rows[0].get("generation_config") if rows else None,
        }
    )
    if output_dir is not None:
        output_path = output_dir / ("%s_%s_metrics.json" % (metrics["model_slug"], inferred_split))
        write_json(output_path, metrics)
    return metrics


def _metric_get(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return ""
        value = value.get(part)
    return value if value is not None else ""


def build_summary_table(metrics_dir: Path, output_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        if path.name == "summary_table.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model_name": payload.get("model_name", ""),
                "split": payload.get("split", ""),
                "dtype": payload.get("dtype", ""),
                "quantization": payload.get("quantization", ""),
                "num_examples": payload.get("num_examples", ""),
                "failure_rate": payload.get("failure_rate", ""),
                "invalid_prediction_rate": payload.get("invalid_prediction_rate", ""),
                "classification_accuracy": _metric_get(payload, "classification.accuracy"),
                "classification_macro_f1": _metric_get(payload, "classification.macro_f1"),
                "classification_weighted_f1": _metric_get(payload, "classification.weighted_f1"),
                "classification_invalid_rate": _metric_get(payload, "classification.invalid_output_rate"),
                "vqa_exact_match": _metric_get(payload, "vqa.exact_match"),
                "vqa_relaxed_accuracy": _metric_get(payload, "vqa.relaxed_accuracy"),
                "vqa_token_f1": _metric_get(payload, "vqa.token_f1"),
                "clarify_accuracy": _metric_get(payload, "clarify_or_respond.clarify_accuracy"),
                "clarify_f1": _metric_get(payload, "clarify_or_respond.clarify_f1"),
                "avg_runtime_seconds": payload.get("avg_runtime_seconds", ""),
                "metrics_path": str(path),
            }
        )
    fieldnames = [
        "model_name",
        "split",
        "dtype",
        "quantization",
        "num_examples",
        "failure_rate",
        "invalid_prediction_rate",
        "classification_accuracy",
        "classification_macro_f1",
        "classification_weighted_f1",
        "classification_invalid_rate",
        "vqa_exact_match",
        "vqa_relaxed_accuracy",
        "vqa_token_f1",
        "clarify_accuracy",
        "clarify_f1",
        "avg_runtime_seconds",
        "metrics_path",
    ]
    write_csv(output_path, rows, fieldnames)
    return rows


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_table) if args.summary_table else output_dir / "summary_table.csv"
    if args.refresh_summary_only:
        rows = build_summary_table(output_dir, summary_path)
        print(json.dumps({"summary_table": str(summary_path), "rows": len(rows)}, indent=2, sort_keys=True))
        return 0
    if not args.predictions:
        raise ValueError("--predictions is required unless --refresh-summary-only is used")
    metrics = evaluate_file(
        Path(args.predictions),
        model_name=args.model_name,
        split=args.split,
        output_dir=output_dir,
    )
    build_summary_table(output_dir, summary_path)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
