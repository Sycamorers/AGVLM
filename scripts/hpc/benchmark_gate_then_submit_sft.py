#!/usr/bin/env python3
"""Submit full SFT only after a benchmark job clears metric gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-job-id", required=True)
    parser.add_argument("--benchmark-output-dir", required=True)
    parser.add_argument("--metrics-path", default="")
    parser.add_argument("--model-key", default="agvlm_phi4_sft_completed")
    parser.add_argument("--min-num-examples", type=int, default=392)
    parser.add_argument("--max-failure-rate", type=float, default=0.0)
    parser.add_argument("--max-invalid-rate", type=float, default=None)
    parser.add_argument("--min-task-macro", type=float, default=None)
    parser.add_argument("--min-vqa-relaxed", type=float, default=None)
    parser.add_argument("--min-clarify-macro-f1", type=float, default=None)
    parser.add_argument("--train-slurm", default="scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm")
    parser.add_argument("--data-config", default="configs/data/sft_train_eval_phi4_max3.yaml")
    parser.add_argument("--model-config", default="configs/model/phi4_reasoning_vision_15b_turin_24g.yaml")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--preflight-config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def slurm_job_completed(job_id: str) -> tuple[bool, dict[str, Any]]:
    result = run_command(["sacct", "-j", job_id, "-X", "--format=JobIDRaw,State,ExitCode,Elapsed", "-P", "-n"])
    payload = {
        "command": " ".join(result.args),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0 or not result.stdout.strip():
        return False, payload
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    first = lines[0].split("|")
    state = first[1] if len(first) > 1 else ""
    exit_code = first[2] if len(first) > 2 else ""
    payload.update({"state": state, "exit_code": exit_code})
    return state == "COMPLETED" and exit_code == "0:0", payload


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


def discover_metrics_path(output_dir: Path, model_key: str) -> Path:
    metrics_dir = output_dir / "metrics"
    candidates = sorted(metrics_dir.glob("*_metrics.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No metrics JSON found under %s" % metrics_dir)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(payload.get("model_key") or "") == model_key:
            return path
    return candidates[0]


def evaluate_gates(metrics: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[dict[str, Any]]]:
    checks = [
        {
            "name": "num_examples",
            "actual": metrics.get("num_examples"),
            "op": ">=",
            "required": args.min_num_examples,
            "passed": int(metrics.get("num_examples") or 0) >= args.min_num_examples,
        },
        {
            "name": "failure_rate",
            "actual": as_float(metrics.get("failure_rate")),
            "op": "<=",
            "required": args.max_failure_rate,
            "passed": (as_float(metrics.get("failure_rate")) or 0.0) <= args.max_failure_rate,
        },
    ]
    optional_checks = [
        ("invalid_prediction_rate", metrics.get("invalid_prediction_rate"), "<=", args.max_invalid_rate),
        ("task_macro_average", metrics.get("task_macro_average"), ">=", args.min_task_macro),
        ("vqa.relaxed_accuracy", metric_get(metrics, "vqa.relaxed_accuracy"), ">=", args.min_vqa_relaxed),
        ("clarify_or_respond.macro_f1", metric_get(metrics, "clarify_or_respond.macro_f1"), ">=", args.min_clarify_macro_f1),
    ]
    for name, actual, op, required in optional_checks:
        if required is None:
            continue
        actual_float = as_float(actual)
        checks.append(
            {
                "name": name,
                "actual": actual_float,
                "op": op,
                "required": required,
                "passed": actual_float is not None and (actual_float <= required if op == "<=" else actual_float >= required),
            }
        )
    return all(bool(check["passed"]) for check in checks), checks


def submit_training(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    export_value = ",".join(
        [
            "ALL",
            "DATA_CONFIG=%s" % args.data_config,
            "MODEL_CONFIG=%s" % args.model_config,
            "TRAIN_CONFIG=%s" % args.train_config,
            "PREFLIGHT_CONFIG=%s" % args.preflight_config,
        ]
    )
    command = ["sbatch", "--parsable", "--export=%s" % export_value, args.train_slurm]
    if args.dry_run:
        return "", {"command": " ".join(command), "dry_run": True}
    result = run_command(command)
    payload = {
        "command": " ".join(command),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        raise RuntimeError("Training submission failed: %s" % payload)
    return result.stdout.strip().split(";")[0], payload


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Benchmark Gate Then SFT Submit",
        "",
        "- Benchmark job: `%s`" % report["benchmark_job_id"],
        "- Decision: **%s**" % report["decision"],
        "- Metrics path: `%s`" % report.get("metrics_path", ""),
        "- Training job: `%s`" % report.get("training_job_id", ""),
        "",
        "| Gate | Actual | Requirement | Pass |",
        "| --- | ---: | ---: | --- |",
    ]
    for check in report.get("checks", []):
        lines.append(
            "| %s | %s | %s %s | %s |"
            % (
                check["name"],
                check.get("actual"),
                check["op"],
                check["required"],
                "yes" if check["passed"] else "no",
            )
        )
    if report.get("reason"):
        lines.extend(["", "Reason: %s" % report["reason"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output: dict[str, Any] = {
        "benchmark_job_id": args.benchmark_job_id,
        "benchmark_output_dir": args.benchmark_output_dir,
        "model_key": args.model_key,
        "train_slurm": args.train_slurm,
        "data_config": args.data_config,
        "model_config": args.model_config,
        "train_config": args.train_config,
        "preflight_config": args.preflight_config,
        "decision": "reject",
        "checks": [],
    }
    completed, slurm_payload = slurm_job_completed(args.benchmark_job_id)
    output["benchmark_slurm"] = slurm_payload
    if not completed:
        output["reason"] = "Benchmark job did not complete successfully."
    else:
        metrics_path = Path(args.metrics_path) if args.metrics_path else discover_metrics_path(Path(args.benchmark_output_dir), args.model_key)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        passed, checks = evaluate_gates(metrics, args)
        output["metrics_path"] = str(metrics_path)
        output["metrics"] = {
            "num_examples": metrics.get("num_examples"),
            "failure_rate": metrics.get("failure_rate"),
            "invalid_prediction_rate": metrics.get("invalid_prediction_rate"),
            "task_macro_average": metrics.get("task_macro_average"),
            "vqa_relaxed_accuracy": metric_get(metrics, "vqa.relaxed_accuracy"),
            "clarify_macro_f1": metric_get(metrics, "clarify_or_respond.macro_f1"),
        }
        output["checks"] = checks
        if passed:
            training_job_id, submit_payload = submit_training(args)
            output["decision"] = "submit_training"
            output["training_job_id"] = training_job_id
            output["training_submission"] = submit_payload
        else:
            output["reason"] = "Benchmark metrics did not pass configured gates."
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), output)
    print(json.dumps({"decision": output["decision"], "training_job_id": output.get("training_job_id", "")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
