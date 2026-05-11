#!/usr/bin/env python3
"""Validate two-stage VLM benchmark readiness without loading models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "vlm_baselines"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from build_phase_splits import build_phase_splits  # noqa: E402
from checkpoint_config import load_model_configurations, validate_all_checkpoint_entries  # noqa: E402
from evaluate_predictions import build_summary_table  # noqa: E402
from metrics import evaluate_prediction_records  # noqa: E402
from prediction_parsing import extract_answer_field, extract_decision_field, extract_structured_sections  # noqa: E402
from utils import ensure_dir, write_json  # noqa: E402


SFT_GUARD_PATTERNS = [
    "scripts/train/train_sft.py",
    "src/agri_vlm/training/sft_trainer.py",
    "configs/train/sft_",
    "scripts/hpc/run_sft_",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["sft", "rl", "both"], default="both")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--output-json", default="reports/benchmark_status_report.json")
    parser.add_argument("--output-markdown", default="reports/benchmark_status_report.md")
    parser.add_argument("--split-dir", default=str(BENCHMARK_ROOT / "splits"))
    parser.add_argument("--model-config", default=str(BENCHMARK_ROOT / "baseline_models.yaml"))
    parser.add_argument("--checkpoint-config", default=str(BENCHMARK_ROOT / "agvlm_checkpoint_models.yaml"))
    return parser.parse_args()


def _check(condition: bool, message: str, *, details: Any = None, severity: str = "error") -> dict[str, Any]:
    return {"ok": bool(condition), "message": message, "severity": severity, "details": details}


def _git_status_paths() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    paths = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def _sft_guard(paths: list[str]) -> dict[str, Any]:
    flagged = [
        path
        for path in paths
        if any(path == pattern or path.startswith(pattern) for pattern in SFT_GUARD_PATTERNS)
    ]
    return {
        "ok": not flagged,
        "message": "No SFT training files/configs/Slurm scripts are dirty." if not flagged else "SFT training-related files are dirty in the worktree.",
        "severity": "warning" if flagged else "info",
        "details": flagged,
    }


def _synthetic_metric_check() -> dict[str, Any]:
    rows = [
        {
            "phase": "sft_benchmark",
            "split": "val",
            "task_type": "classification",
            "verifier_mode": "label",
            "raw_output": "Answer: tomato late blight",
            "parsed_prediction": "tomato late blight",
            "normalized_prediction": "tomato late blight",
            "parse_status": "exact",
            "invalid_prediction": False,
            "ground_truth": "tomato late blight",
            "references": ["tomato late blight"],
            "source_dataset": "synthetic",
        },
        {
            "phase": "sft_benchmark",
            "split": "val",
            "task_type": "vqa",
            "verifier_mode": "exact_match",
            "raw_output": "Answer: Yes",
            "parsed_prediction": "Yes",
            "normalized_prediction": "yes",
            "parse_status": "exact",
            "invalid_prediction": False,
            "ground_truth": "Yes",
            "references": ["Yes"],
            "source_dataset": "synthetic",
        },
        {
            "phase": "rl_benchmark",
            "split": "val",
            "task_type": "clarify_or_respond",
            "verifier_mode": "clarify",
            "raw_output": "Decision: clarify\nAnswer: Please provide a closer leaf image.",
            "parsed_prediction": "clarify",
            "normalized_prediction": "clarify",
            "parse_status": "exact",
            "invalid_prediction": False,
            "ground_truth": "clarify",
            "references": ["clarify"],
            "source_dataset": "synthetic",
        },
        {
            "phase": "rl_benchmark",
            "split": "val",
            "task_type": "consultation",
            "verifier_mode": "structured",
            "raw_output": "Diagnosis: possible blight\nEvidence: lesions\nUncertainty: confirm in field\nManagement: remove debris\nFollow-up: send close-up?",
            "parsed_prediction": "structured",
            "normalized_prediction": "structured",
            "parse_status": "exact",
            "invalid_prediction": False,
            "ground_truth": "possible blight",
            "references": ["possible blight"],
            "source_dataset": "synthetic",
            "verifier": {
                "required_sections": ["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
                "management_keywords": ["remove debris"],
                "forbidden_claims": ["guaranteed cure"],
                "uncertainty_required": True,
            },
        },
    ]
    metrics = evaluate_prediction_records(rows)
    ok = (
        metrics["classification"]["top1_accuracy"] == 1.0
        and metrics["short_vqa"]["yes_no_accuracy"] == 1.0
        and metrics["clarify_or_respond"]["decision_accuracy"] == 1.0
        and metrics["consultation"]["structured_section_compliance"] == 1.0
    )
    return _check(ok, "Metrics module can score synthetic benchmark predictions.", details=metrics)


def _parser_check() -> dict[str, Any]:
    answer, answer_status = extract_answer_field("Answer: tomato late blight")
    decision, decision_status = extract_decision_field("Decision: clarify\nAnswer: Need a close-up.")
    sections = extract_structured_sections("Diagnosis: blight\nEvidence: spots\nManagement: prune")
    ok = answer == "tomato late blight" and answer_status == "exact" and decision == "clarify" and decision_status == "exact" and "diagnosis" in sections
    return _check(ok, "Prediction parser handles Answer, Decision, and structured sections.")


def _summary_check() -> dict[str, Any]:
    path = BENCHMARK_ROOT / "results" / "metrics" / "summary_table.csv"
    try:
        rows = build_summary_table(path.parent, path)
    except Exception as exc:
        return _check(False, "Summary table build failed.", details="%s: %s" % (type(exc).__name__, exc))
    return _check(True, "Summary table can be refreshed.", details={"rows": len(rows), "path": str(path)})


def _slurm_check() -> dict[str, Any]:
    required = [
        BENCHMARK_ROOT / "slurm" / "run_sft_benchmark_24gb.sbatch",
        BENCHMARK_ROOT / "slurm" / "run_sft_benchmark_agvlm_checkpoint.sbatch",
        BENCHMARK_ROOT / "slurm" / "run_rl_benchmark_24gb.sbatch",
        BENCHMARK_ROOT / "slurm" / "run_rl_benchmark_agvlm_checkpoint.sbatch",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return _check(not missing, "Required benchmark Slurm scripts exist.", details={"missing": missing})


def _docs_check() -> dict[str, Any]:
    required = [
        "docs/project_overview.md",
        "docs/project_plan.md",
        "docs/progress_tracker.md",
        "docs/experiment_roadmap.md",
        "docs/benchmark_plan.md",
        "docs/eval_plan.md",
        "docs/results_artifacts.md",
        "docs/session_handoff.md",
        "reports/benchmark_framework_audit.md",
    ]
    missing = [path for path in required if not (REPO_ROOT / path).exists()]
    return _check(not missing, "Required benchmark/project docs exist.", details={"missing": missing}, severity="warning" if missing else "info")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    split_report = build_phase_splits(
        phase=args.phase,
        output_dir=Path(args.split_dir),
        seed=42,
        force=False,
        allow_fallback_split=False,
        write_report=True,
    )
    phases = split_report.get("phases") or {}
    for phase_name, payload in phases.items():
        overlap = payload.get("train_eval_overlap") or {}
        checks.append(
            _check(
                overlap.get("exact_sample_id_count", 0) == 0 and overlap.get("group_key_count", 0) == 0,
                "%s has no train/eval sample-id or group overlap." % phase_name,
                details=overlap,
            )
        )
        checks.append(
            _check(
                payload.get("duplicate_sample_id_count", 0) == 0,
                "%s split manifests have no duplicate sample IDs." % phase_name,
                details=payload.get("duplicate_sample_id_examples", []),
            )
        )
    model_entries = load_model_configurations(
        model_config_path=Path(args.model_config),
        checkpoint_config_path=Path(args.checkpoint_config),
    )
    external_keys = sorted(
        key
        for key, entry in model_entries.items()
        if entry.get("checkpoint_type") == "external_baseline" and key == entry.get("model_key")
    )
    checks.append(_check(bool(external_keys), "External baseline model config parses.", details=external_keys))
    checkpoint_validation = validate_all_checkpoint_entries(Path(args.checkpoint_config), phase=args.phase)
    checks.append(
        _check(
            checkpoint_validation.get("exists", False),
            "AGVLM checkpoint config parses; placeholder paths are warnings until selected for a run.",
            details=checkpoint_validation,
            severity="warning" if checkpoint_validation.get("warnings") else "info",
        )
    )
    checks.append(_parser_check())
    checks.append(_synthetic_metric_check())
    checks.append(_summary_check())
    checks.append(_slurm_check())
    checks.append(_docs_check())
    dirty_paths = _git_status_paths()
    checks.append(_sft_guard(dirty_paths))
    errors = [check for check in checks if not check["ok"] and check["severity"] == "error"]
    warnings = [check for check in checks if (not check["ok"] and check["severity"] != "error") or (check["ok"] and check["severity"] == "warning")]
    return {
        "ok": not errors,
        "phase": args.phase,
        "checks": checks,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "split_report_path": str(Path(args.split_dir) / "benchmark_split_report.json"),
        "dirty_paths": dirty_paths,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Benchmark Status Report",
        "",
        "- phase: `%s`" % report.get("phase"),
        "- overall ok: `%s`" % report.get("ok"),
        "- errors: `%s`" % report.get("error_count"),
        "- warnings: `%s`" % report.get("warning_count"),
        "",
        "| Status | Severity | Check |",
        "| --- | --- | --- |",
    ]
    for check in report.get("checks") or []:
        lines.append(
            "| %s | %s | %s |"
            % ("ok" if check.get("ok") else "fail", check.get("severity"), str(check.get("message")).replace("|", "\\|"))
        )
    lines.extend(
        [
            "",
            "## Dirty SFT Guard",
            "",
            "The status check reports dirty SFT training files if present, but it does not revert user work.",
            "",
            "```text",
            "\n".join(report.get("dirty_paths") or []),
            "```",
            "",
        ]
    )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.write_report:
        write_json(REPO_ROOT / args.output_json, report)
        write_markdown(report, REPO_ROOT / args.output_markdown)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
