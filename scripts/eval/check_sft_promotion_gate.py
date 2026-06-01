#!/usr/bin/env python3
"""Apply a hard SFT promotion gate to multi-model inference metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_METRICS = {
    "task_macro_average": "higher",
    "short_vqa.relaxed_accuracy": "higher",
    "clarify_or_respond.macro_f1": "higher",
    "num_invalid_predictions": "lower",
}
DIAGNOSTIC_METRICS = {
    "classification.top1_accuracy": "higher",
    "classification.macro_f1": "higher",
    "local_metrics.average_reward": "higher",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True, help="multi_model_metrics.json from SFT comparison reporting.")
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--baseline-key", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--task-macro-margin", type=float, default=0.0)
    parser.add_argument("--vqa-margin", type=float, default=0.0)
    parser.add_argument("--clarify-margin", type=float, default=0.0)
    parser.add_argument("--invalid-margin", type=int, default=0)
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser.parse_args()


def metric_get(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "%.6f" % value
    if value is None:
        return ""
    return str(value)


def evaluate_metric(
    *,
    name: str,
    direction: str,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    margin: float,
) -> dict[str, Any]:
    candidate_value = metric_get(candidate, name)
    baseline_value = metric_get(baseline, name)
    if name == "num_invalid_predictions":
        candidate_number = as_int(candidate_value)
        baseline_number = as_int(baseline_value)
    else:
        candidate_number = as_float(candidate_value)
        baseline_number = as_float(baseline_value)
    if candidate_number is None or baseline_number is None:
        passed = False
        delta = None
    elif direction == "higher":
        delta = float(candidate_number) - float(baseline_number)
        passed = delta >= margin
    else:
        delta = float(baseline_number) - float(candidate_number)
        passed = delta >= margin
    return {
        "metric": name,
        "direction": direction,
        "candidate": candidate_number,
        "baseline": baseline_number,
        "delta_in_preferred_direction": delta,
        "margin": margin,
        "passed": passed,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# SFT Promotion Gate",
        "",
        "- Candidate: `%s`" % report["candidate_key"],
        "- Baseline: `%s`" % report["baseline_key"],
        "- Decision: **%s**" % ("PASS" if report["passed"] else "REJECT"),
        "",
        "| Required Metric | Baseline | Candidate | Preferred Delta | Required Margin | Pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["required_metrics"]:
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                row["metric"],
                fmt(row["baseline"]),
                fmt(row["candidate"]),
                fmt(row["delta_in_preferred_direction"]),
                fmt(row["margin"]),
                "yes" if row["passed"] else "no",
            )
        )
    lines.extend(["", "| Diagnostic Metric | Baseline | Candidate | Preferred Delta |", "| --- | ---: | ---: | ---: |"])
    for row in report["diagnostic_metrics"]:
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                row["metric"],
                fmt(row["baseline"]),
                fmt(row["candidate"]),
                fmt(row["delta_in_preferred_direction"]),
            )
        )
    if report["failed_required_metrics"]:
        lines.extend(["", "Failed required metrics:"])
        for metric_name in report["failed_required_metrics"]:
            lines.append("- `%s`" % metric_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    models = payload.get("models") or {}
    if args.candidate_key not in models:
        raise KeyError("Candidate model key not found: %s" % args.candidate_key)
    if args.baseline_key not in models:
        raise KeyError("Baseline model key not found: %s" % args.baseline_key)
    candidate = models[args.candidate_key]
    baseline = models[args.baseline_key]
    margins = {
        "task_macro_average": args.task_macro_margin,
        "short_vqa.relaxed_accuracy": args.vqa_margin,
        "clarify_or_respond.macro_f1": args.clarify_margin,
        "num_invalid_predictions": args.invalid_margin,
    }
    required_rows = [
        evaluate_metric(
            name=name,
            direction=direction,
            candidate=candidate,
            baseline=baseline,
            margin=margins[name],
        )
        for name, direction in REQUIRED_METRICS.items()
    ]
    diagnostic_rows = [
        evaluate_metric(name=name, direction=direction, candidate=candidate, baseline=baseline, margin=0.0)
        for name, direction in DIAGNOSTIC_METRICS.items()
    ]
    failed = [row["metric"] for row in required_rows if not row["passed"]]
    report = {
        "metrics_json": str(Path(args.metrics_json)),
        "candidate_key": args.candidate_key,
        "baseline_key": args.baseline_key,
        "passed": not failed,
        "required_metrics": required_rows,
        "diagnostic_metrics": diagnostic_rows,
        "failed_required_metrics": failed,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), report)
    print(json.dumps({"passed": report["passed"], "failed_required_metrics": failed}, indent=2, sort_keys=True))
    return 1 if failed and args.fail_on_reject else 0


if __name__ == "__main__":
    raise SystemExit(main())
