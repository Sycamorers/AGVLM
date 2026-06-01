#!/usr/bin/env python3
"""Build a static dashboard for the current SFT benchmark decision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "vlm_baselines"
sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluate_predictions import refresh_parse_fields  # noqa: E402

ACTIVE_BASELINE_KEY = "agvlm_phi4_sft_completed"
BASE_MODEL_KEY = "agvlm_phi4_base"
BALANCED_V2_KEY = "agvlm_phi4_sft_balanced_v2_instructional_completed"
PILOT_KEY = "agvlm_phi4_sft_classification_repair_instructional_pilot_completed"
STAGE2_KEY = "agvlm_phi4_sft_classification_repair_instructional_stage2_b200_candidate"

MODEL_LABELS = {
    BASE_MODEL_KEY: "Phi-4 Base",
    ACTIVE_BASELINE_KEY: "Active Previous SFT",
    BALANCED_V2_KEY: "Balanced-v2 SFT",
    PILOT_KEY: "Classification-Repair Pilot",
    STAGE2_KEY: "Stage2 B200 Candidate",
}

GENERIC_OUTPUTS = {
    "",
    "answer",
    "answer:",
    "plant",
    "plant.",
    "plant disease",
    "plant disease.",
    "disease",
    "disease.",
    "unknown",
    "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-output-dir",
        default="benchmarks/vlm_baselines/results/agvlm_stage2_b200_benchmark_20260601",
    )
    parser.add_argument(
        "--pilot-output-dir",
        default="benchmarks/vlm_baselines/results/agvlm_classification_repair_pilot_benchmark_promptfix_20260531",
    )
    parser.add_argument(
        "--comparison-json",
        default="reports/sft_regression_audit/completed_sft_benchmark_comparison_20260519.json",
    )
    parser.add_argument(
        "--adapter-validation",
        default="/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-classification-repair-instructional-stage2-b200-4gpu/adapter_validation.json",
    )
    parser.add_argument(
        "--training-metrics",
        default="outputs/sft/phi4-reasoning-vision-15b-classification-repair-instructional-stage2-b200-4gpu/metrics/train_metrics.jsonl",
    )
    parser.add_argument(
        "--slurm-out",
        default="benchmarks/vlm_baselines/logs/slurm/agri-sft-bench-33629150.out",
    )
    parser.add_argument(
        "--slurm-err",
        default="benchmarks/vlm_baselines/logs/slurm/agri-sft-bench-33629150.err",
    )
    parser.add_argument("--output-dir", default="reports/sft_stage_decision_20260601")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_get(payload: dict[str, Any] | None, dotted: str) -> Any:
    value: Any = payload or {}
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


def fmt(value: Any, *, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return "n/a" if value in (None, "") else str(value)
    if abs(number) >= 100:
        return "%.0f" % number
    return ("%." + str(digits) + "f") % number


def pct(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else "%.1f%%" % (number * 100.0)


def load_metrics_from_output_dir(output_dir: Path) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "metrics").glob("*_metrics.json")):
        payload = read_json(path)
        key = str(payload.get("model_key") or path.stem)
        payload["_metrics_path"] = str(path)
        metrics[key] = payload
    return metrics


def load_reference_metrics(comparison_json: Path, pilot_output_dir: Path) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    if comparison_json.exists():
        payload = read_json(comparison_json)
        for key, model_metrics in (payload.get("models") or {}).items():
            metrics[str(key)] = model_metrics
    metrics.update(load_metrics_from_output_dir(pilot_output_dir))
    return metrics


def prediction_files(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "predictions").glob("*.jsonl"))


def normalized_raw(text: str | None) -> str:
    return " ".join(str(text or "").strip().lower().split())


def answer_body(raw_output: str) -> str:
    lowered = raw_output.strip()
    if lowered.lower().startswith("answer:"):
        return lowered.split(":", 1)[1].strip()
    return lowered


def is_format_like_failure(record: dict[str, Any]) -> bool:
    raw = str(record.get("raw_output") or "")
    normalized = normalized_raw(raw)
    parsed = normalized_raw(record.get("parsed_prediction"))
    status = str(record.get("parse_status") or "")
    if not raw.strip() or normalized in GENERIC_OUTPUTS or parsed in GENERIC_OUTPUTS:
        return True
    if status in {"failed", "ambiguous", "missing"}:
        return True
    if normalized.startswith("answer:") and not answer_body(raw):
        return True
    if record.get("task_type") == "clarify_or_respond" and status != "exact":
        return True
    return False


def has_output_contract_issue(record: dict[str, Any]) -> bool:
    status = str(record.get("parse_status") or "")
    task_type = str(record.get("task_type") or "")
    verifier_mode = str(record.get("verifier_mode") or "")
    raw = str(record.get("raw_output") or "")
    if is_format_like_failure(record):
        return True
    if verifier_mode == "label" or task_type in {"classification", "label_diagnosis"}:
        return status != "exact"
    if task_type == "vqa" or verifier_mode in {"exact_match", "synonym"}:
        return status != "exact"
    if task_type == "clarify_or_respond" or verifier_mode == "clarify":
        return status != "exact" or "decision:" not in raw.lower()
    if task_type == "consultation" or verifier_mode == "structured":
        return status not in {"exact", "raw"}
    return False


def output_diagnostics_for_file(path: Path) -> dict[str, Any]:
    rows = refresh_parse_fields(read_jsonl(path))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_type") or "missing")].append(row)
    parse_status = Counter(str(row.get("parse_status") or "missing") for row in rows)
    invalid = [row for row in rows if row.get("invalid_prediction")]
    out_of_label_space = [row for row in rows if row.get("out_of_label_space") or row.get("parse_status") == "out_of_label_space"]
    format_like = [row for row in invalid if is_format_like_failure(row)]
    contract_issues = [row for row in rows if has_output_contract_issue(row)]
    empty = [row for row in rows if not str(row.get("raw_output") or "").strip()]
    answer_only = [row for row in rows if normalized_raw(row.get("raw_output")) in {"answer:", "answer"}]
    explicit_decisions = [
        row
        for row in rows
        if row.get("task_type") == "clarify_or_respond"
        and "decision:" in str(row.get("raw_output") or "").lower()
    ]
    clarify_rows = by_task.get("clarify_or_respond", [])
    task_rows = {}
    for task, items in sorted(by_task.items()):
        task_invalid = [row for row in items if row.get("invalid_prediction")]
        task_out_of_label_space = [
            row for row in items if row.get("out_of_label_space") or row.get("parse_status") == "out_of_label_space"
        ]
        task_rows[task] = {
            "num_examples": len(items),
            "invalid_count": len(task_invalid),
            "invalid_rate": len(task_invalid) / float(len(items)) if items else None,
            "parse_status_counts": dict(Counter(str(row.get("parse_status") or "missing") for row in items)),
            "format_like_invalid_count": sum(1 for row in task_invalid if is_format_like_failure(row)),
            "format_contract_issue_count": sum(1 for row in items if has_output_contract_issue(row)),
            "out_of_label_space_count": len(task_out_of_label_space),
        }
    prefix_counts = Counter(normalized_raw(row.get("raw_output"))[:80] for row in rows)
    examples = []
    for row in invalid[:40]:
        examples.append(
            {
                "sample_id": row.get("sample_id"),
                "task_type": row.get("task_type"),
                "source_dataset": row.get("source_dataset"),
                "ground_truth": row.get("ground_truth"),
                "parse_status": row.get("parse_status"),
                "parsed_prediction": row.get("parsed_prediction"),
                "raw_output": str(row.get("raw_output") or "")[:360],
                "format_like_failure": is_format_like_failure(row),
            }
        )
    contract_examples = []
    for row in contract_issues[:40]:
        contract_examples.append(
            {
                "sample_id": row.get("sample_id"),
                "task_type": row.get("task_type"),
                "source_dataset": row.get("source_dataset"),
                "ground_truth": row.get("ground_truth"),
                "parse_status": row.get("parse_status"),
                "parsed_prediction": row.get("parsed_prediction"),
                "raw_output": str(row.get("raw_output") or "")[:360],
            }
        )
    return {
        "prediction_path": str(path),
        "model_key": str(rows[0].get("model_key") if rows else path.stem),
        "num_examples": len(rows),
        "invalid_count": len(invalid),
        "invalid_rate": len(invalid) / float(len(rows)) if rows else None,
        "out_of_label_space_count": len(out_of_label_space),
        "out_of_label_space_rate": len(out_of_label_space) / float(len(rows)) if rows else None,
        "empty_output_count": len(empty),
        "answer_only_count": len(answer_only),
        "format_like_invalid_count": len(format_like),
        "format_like_invalid_rate": len(format_like) / float(len(rows)) if rows else None,
        "format_contract_issue_count": len(contract_issues),
        "format_contract_issue_rate": len(contract_issues) / float(len(rows)) if rows else None,
        "parse_status_counts": dict(parse_status),
        "by_task": task_rows,
        "clarify_explicit_decision_rate": len(explicit_decisions) / float(len(clarify_rows)) if clarify_rows else None,
        "common_output_prefixes": prefix_counts.most_common(10),
        "invalid_examples": examples,
        "format_contract_examples": contract_examples,
    }


def load_prediction_diagnostics(*output_dirs: Path) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for output_dir in output_dirs:
        for path in prediction_files(output_dir):
            payload = output_diagnostics_for_file(path)
            diagnostics[payload["model_key"]] = payload
    return diagnostics


def evaluate_gate(candidate: dict[str, Any] | None, baseline: dict[str, Any] | None) -> dict[str, Any]:
    checks = [
        ("task_macro_average", "higher", 0.0),
        ("short_vqa.relaxed_accuracy", "higher", 0.0),
        ("clarify_or_respond.macro_f1", "higher", 0.0),
        ("num_invalid_predictions", "lower", 0.0),
    ]
    rows = []
    for metric, direction, margin in checks:
        candidate_value = metric_get(candidate, metric)
        baseline_value = metric_get(baseline, metric)
        if metric == "num_invalid_predictions":
            cand = as_int(candidate_value)
            base = as_int(baseline_value)
        else:
            cand = as_float(candidate_value)
            base = as_float(baseline_value)
        if cand is None or base is None:
            delta = None
            passed = False
        elif direction == "higher":
            delta = float(cand) - float(base)
            passed = delta >= margin
        else:
            delta = float(base) - float(cand)
            passed = delta >= margin
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "candidate": cand,
                "baseline": base,
                "delta_in_preferred_direction": delta,
                "passed": passed,
            }
        )
    return {"passed": all(row["passed"] for row in rows), "checks": rows}


def stage_status(stage_output_dir: Path, stage_metrics: dict[str, dict[str, Any]], slurm_out: Path, slurm_err: Path) -> str:
    if STAGE2_KEY in stage_metrics:
        return "completed"
    if any(prediction_files(stage_output_dir)):
        return "running_or_partial"
    if slurm_out.exists() or slurm_err.exists():
        return "submitted_or_running"
    return "not_run"


def training_summary(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    rows = read_jsonl(metrics_path)
    losses = [as_float(row.get("loss")) for row in rows if as_float(row.get("loss")) is not None]
    eval_losses = [as_float(row.get("eval_loss")) for row in rows if as_float(row.get("eval_loss")) is not None]
    steps = [as_int(row.get("step")) for row in rows if as_int(row.get("step")) is not None]
    return {
        "path": str(metrics_path),
        "num_metric_rows": len(rows),
        "max_step": max(steps) if steps else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_mean": mean(losses) if losses else None,
        "eval_loss_first": eval_losses[0] if eval_losses else None,
        "eval_loss_last": eval_losses[-1] if eval_losses else None,
    }


def slurm_notes(out_path: Path, err_path: Path) -> dict[str, Any]:
    notes: dict[str, Any] = {
        "out_path": str(out_path),
        "err_path": str(err_path),
        "out_exists": out_path.exists(),
        "err_exists": err_path.exists(),
    }
    if out_path.exists():
        text = out_path.read_text(encoding="utf-8", errors="replace")
        notes["completed_samples_markers"] = [
            line.strip() for line in text.splitlines() if line.strip().startswith("completed ")
        ][-5:]
    if err_path.exists():
        err = err_path.read_text(encoding="utf-8", errors="replace")
        notes["hf_remote_code_refresh_warning"] = "new version of the following files was downloaded" in err.lower()
        notes["error_tail"] = "\n".join(err.splitlines()[-20:])
    return notes


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    stage_output_dir = (REPO_ROOT / args.stage_output_dir).resolve()
    pilot_output_dir = (REPO_ROOT / args.pilot_output_dir).resolve()
    comparison_json = (REPO_ROOT / args.comparison_json).resolve()
    stage_metrics = load_metrics_from_output_dir(stage_output_dir)
    reference_metrics = load_reference_metrics(comparison_json, pilot_output_dir)
    all_metrics = {**reference_metrics, **stage_metrics}
    diagnostics = load_prediction_diagnostics(pilot_output_dir, stage_output_dir)
    adapter_path = Path(args.adapter_validation)
    adapter_validation = read_json(adapter_path) if adapter_path.exists() else {}
    train = training_summary((REPO_ROOT / args.training_metrics).resolve())
    slurm = slurm_notes((REPO_ROOT / args.slurm_out).resolve(), (REPO_ROOT / args.slurm_err).resolve())
    candidate = all_metrics.get(STAGE2_KEY)
    active = all_metrics.get(ACTIVE_BASELINE_KEY)
    pilot = all_metrics.get(PILOT_KEY)
    active_gate = evaluate_gate(candidate, active) if candidate and active else {"passed": False, "checks": []}
    pilot_check = evaluate_gate(candidate, pilot) if candidate and pilot else {"passed": False, "checks": []}
    status = stage_status(stage_output_dir, stage_metrics, Path(args.slurm_out), Path(args.slurm_err))
    if status != "completed":
        decision = "WAIT_FOR_BENCHMARK"
    elif active_gate["passed"] and pilot_check["passed"]:
        decision = "PROMOTE"
    elif active_gate["passed"]:
        decision = "PASS_ACTIVE_BASELINE_REVIEW_PILOT_REGRESSION"
    else:
        decision = "DO_NOT_PROMOTE"
    return {
        "stage_status": status,
        "decision": decision,
        "stage_output_dir": str(stage_output_dir),
        "models": all_metrics,
        "output_diagnostics": diagnostics,
        "active_gate": active_gate,
        "pilot_regression_check": pilot_check,
        "adapter_validation": adapter_validation,
        "training_summary": train,
        "slurm_notes": slurm,
        "paths": {
            "comparison_json": str(comparison_json),
            "pilot_output_dir": str(pilot_output_dir),
            "adapter_validation": str(adapter_path),
        },
    }


def model_metric_row(model_key: str, metrics: dict[str, Any], diagnostics: dict[str, Any] | None) -> str:
    invalid = metrics.get("num_invalid_predictions")
    if invalid is None and diagnostics:
        invalid = diagnostics.get("invalid_count")
    format_like = diagnostics.get("format_like_invalid_count") if diagnostics else None
    contract_issues = diagnostics.get("format_contract_issue_count") if diagnostics else None
    out_of_label_space = metric_get(metrics, "classification.out_of_label_space_rate")
    return (
        "<tr>"
        f"<th>{esc(MODEL_LABELS.get(model_key, model_key))}</th>"
        f"<td>{esc(model_key)}</td>"
        f"<td>{fmt(metrics.get('num_examples'))}</td>"
        f"<td>{fmt(metrics.get('task_macro_average'))}</td>"
        f"<td>{fmt(metric_get(metrics, 'classification.macro_f1'))}</td>"
        f"<td>{fmt(metric_get(metrics, 'short_vqa.relaxed_accuracy'))}</td>"
        f"<td>{fmt(metric_get(metrics, 'clarify_or_respond.macro_f1'))}</td>"
        f"<td>{fmt(invalid)}</td>"
        f"<td>{pct(metrics.get('invalid_prediction_rate') if metrics.get('invalid_prediction_rate') is not None else (diagnostics or {}).get('invalid_rate'))}</td>"
        f"<td>{fmt(format_like)}</td>"
        f"<td>{fmt(contract_issues)}</td>"
        f"<td>{pct(out_of_label_space)}</td>"
        "</tr>"
    )


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def bar(value: Any, *, max_value: float = 1.0) -> str:
    number = as_float(value)
    if number is None:
        width = 0
    else:
        width = max(0, min(100, int(round(100 * number / max_value))))
    return f'<div class="bar"><span style="width:{width}%"></span></div>'


def gate_table(title: str, gate: dict[str, Any]) -> str:
    rows = []
    for row in gate.get("checks") or []:
        cls = "pass" if row.get("passed") else "fail"
        rows.append(
            "<tr>"
            f"<th>{esc(row['metric'])}</th>"
            f"<td>{esc(row['direction'])}</td>"
            f"<td>{fmt(row.get('baseline'))}</td>"
            f"<td>{fmt(row.get('candidate'))}</td>"
            f"<td>{fmt(row.get('delta_in_preferred_direction'))}</td>"
            f"<td class='{cls}'>{'pass' if row.get('passed') else 'fail'}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='6'>No candidate benchmark metrics available yet.</td></tr>")
    return (
        f"<section><h2>{esc(title)}</h2><table>"
        "<thead><tr><th>Metric</th><th>Direction</th><th>Baseline</th><th>Candidate</th><th>Preferred Delta</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def diagnostics_section(diagnostics: dict[str, dict[str, Any]]) -> str:
    cards = []
    for model_key, payload in sorted(diagnostics.items()):
        by_task_rows = []
        for task, task_payload in (payload.get("by_task") or {}).items():
            by_task_rows.append(
                "<tr>"
                f"<th>{esc(task)}</th>"
                f"<td>{fmt(task_payload.get('num_examples'))}</td>"
                f"<td>{fmt(task_payload.get('invalid_count'))}</td>"
                f"<td>{pct(task_payload.get('invalid_rate'))}</td>"
                f"<td>{fmt(task_payload.get('format_like_invalid_count'))}</td>"
                f"<td>{fmt(task_payload.get('format_contract_issue_count'))}</td>"
                f"<td>{fmt(task_payload.get('out_of_label_space_count'))}</td>"
                f"<td>{esc(task_payload.get('parse_status_counts'))}</td>"
                "</tr>"
            )
        examples = []
        for example in (payload.get("invalid_examples") or [])[:6]:
            examples.append(
                "<li>"
                f"<strong>{esc(example.get('task_type'))}</strong> "
                f"<code>{esc(example.get('sample_id'))}</code> "
                f"status={esc(example.get('parse_status'))}, parsed={esc(example.get('parsed_prediction'))}<br>"
                f"<span>{esc(example.get('raw_output'))}</span>"
                "</li>"
            )
        cards.append(
            "<article class='card wide'>"
            f"<h3>{esc(MODEL_LABELS.get(model_key, model_key))}</h3>"
            f"<p><code>{esc(payload.get('prediction_path'))}</code></p>"
            "<div class='kpis'>"
            f"<div><b>{fmt(payload.get('invalid_count'))}</b><span>Invalid</span></div>"
            f"<div><b>{pct(payload.get('invalid_rate'))}</b><span>Invalid rate</span></div>"
            f"<div><b>{fmt(payload.get('format_like_invalid_count'))}</b><span>Format-like invalid</span></div>"
            f"<div><b>{fmt(payload.get('format_contract_issue_count'))}</b><span>Format contract issue</span></div>"
            f"<div><b>{fmt(payload.get('out_of_label_space_count'))}</b><span>Out-of-label</span></div>"
            f"<div><b>{pct(payload.get('clarify_explicit_decision_rate'))}</b><span>Explicit Decision rate</span></div>"
            "</div>"
            "<table><thead><tr><th>Task</th><th>N</th><th>Invalid</th><th>Invalid Rate</th><th>Format-like</th><th>Contract Issue</th><th>Out-of-label</th><th>Parse Status</th></tr></thead>"
            f"<tbody>{''.join(by_task_rows)}</tbody></table>"
            f"<h4>Invalid examples</h4><ul>{''.join(examples) if examples else '<li>No invalid examples captured.</li>'}</ul>"
            "</article>"
        )
    return "<section><h2>Output Engineering Diagnostics</h2><div class='cards'>%s</div></section>" % "".join(cards)


def render_html(summary: dict[str, Any]) -> str:
    models = summary["models"]
    diagnostics = summary["output_diagnostics"]
    order = [BASE_MODEL_KEY, ACTIVE_BASELINE_KEY, BALANCED_V2_KEY, PILOT_KEY, STAGE2_KEY]
    metric_rows = []
    for key in order:
        if key in models:
            metric_rows.append(model_metric_row(key, models[key], diagnostics.get(key)))
    status = summary["stage_status"]
    decision = summary["decision"]
    candidate = models.get(STAGE2_KEY)
    active = models.get(ACTIVE_BASELINE_KEY)
    task_macro = metric_get(candidate, "task_macro_average")
    active_macro = metric_get(active, "task_macro_average")
    adapter = summary.get("adapter_validation") or {}
    train = summary.get("training_summary") or {}
    slurm = summary.get("slurm_notes") or {}
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AGVLM SFT Stage Decision</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8f3;
      --ink: #1f2521;
      --muted: #647067;
      --line: #d9dfd7;
      --panel: #ffffff;
      --accent: #2f7661;
      --warn: #b15d2a;
      --bad: #a23b3b;
      --good: #23724d;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }}
    header {{ padding: 28px 32px 20px; background: #20372e; color: white; }}
    main {{ padding: 24px 32px 40px; max-width: 1480px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 30px 0 12px; font-size: 19px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; letter-spacing: 0; }}
    h4 {{ margin: 18px 0 8px; }}
    p {{ margin: 6px 0; color: var(--muted); }}
    code {{ background: #eef2ec; border: 1px solid var(--line); border-radius: 4px; padding: 1px 4px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 12px; margin-top: 18px; }}
    .tile, .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(20, 32, 24, .04); }}
    .tile b {{ display: block; font-size: 24px; }}
    .tile span {{ color: var(--muted); }}
    .decision {{ color: white; background: {('#23724d' if decision == 'PROMOTE' else '#b15d2a' if decision.startswith('WAIT') or 'REVIEW' in decision else '#a23b3b')}; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 9px 10px; vertical-align: top; }}
    thead th {{ background: #eef2ec; font-size: 12px; color: #47524b; text-transform: uppercase; }}
    tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
    .pass {{ color: var(--good); font-weight: 700; }}
    .fail {{ color: var(--bad); font-weight: 700; }}
    .bar {{ height: 8px; background: #edf0ea; border-radius: 999px; overflow: hidden; min-width: 90px; }}
    .bar span {{ display: block; height: 100%; background: var(--accent); }}
    .cards {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .wide {{ min-width: 0; overflow: auto; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin: 10px 0 14px; }}
    .kpis div {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fbfcfa; }}
    .kpis b {{ display: block; font-size: 20px; }}
    .kpis span {{ color: var(--muted); font-size: 12px; }}
    ul {{ margin: 8px 0 0; padding-left: 18px; }}
    li {{ margin: 8px 0; }}
    li span {{ color: var(--muted); }}
    .note {{ background: #fff8ee; border: 1px solid #ead7bd; border-radius: 8px; padding: 12px; }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .summary, .cards, .kpis {{ grid-template-columns: 1fr; }}
      table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>AGVLM SFT Stage Decision</h1>
    <p>Ground-level RGB agriculture consultation benchmark gate for the current Stage2 B200 SFT candidate.</p>
  </header>
  <main>
    <section class="summary">
      <div class="tile"><b>{esc(status)}</b><span>Benchmark status</span></div>
      <div class="tile decision"><b>{esc(decision)}</b><span>Decision</span></div>
      <div class="tile"><b>{fmt(task_macro)}</b><span>Stage2 task macro</span>{bar(task_macro)}</div>
      <div class="tile"><b>{fmt(active_macro)}</b><span>Active SFT task macro</span>{bar(active_macro)}</div>
    </section>

    <section>
      <h2>Benchmark Metrics</h2>
      <table>
        <thead>
          <tr><th>Model</th><th>Key</th><th>N</th><th>Task Macro</th><th>Class F1</th><th>VQA Relaxed</th><th>Clarify F1</th><th>Invalid</th><th>Invalid Rate</th><th>Format-like Invalid</th><th>Contract Issues</th><th>Class Out-of-label</th></tr>
        </thead>
        <tbody>{''.join(metric_rows)}</tbody>
      </table>
    </section>

    {gate_table('Promotion Gate: Stage2 vs Active Previous SFT', summary['active_gate'])}
    {gate_table('Regression Check: Stage2 vs Classification-Repair Pilot', summary['pilot_regression_check'])}

    <section>
      <h2>Training and Adapter Checks</h2>
      <div class="cards">
        <article class="card">
          <h3>Adapter Validation</h3>
          <p><code>{esc(adapter.get('adapter_dir'))}</code></p>
          <table><tbody>
            <tr><th>Format</th><td>{esc(adapter.get('format'))}</td></tr>
            <tr><th>PEFT Type</th><td>{esc(adapter.get('peft_type'))}</td></tr>
            <tr><th>Num tensors</th><td>{fmt(adapter.get('num_tensors'))}</td></tr>
            <tr><th>Non-empty tensors</th><td>{fmt(adapter.get('non_empty_tensors'))}</td></tr>
            <tr><th>First tensor</th><td>{esc(adapter.get('first_tensor'))}</td></tr>
          </tbody></table>
        </article>
        <article class="card">
          <h3>Training Metrics</h3>
          <p><code>{esc(train.get('path'))}</code></p>
          <table><tbody>
            <tr><th>Max step</th><td>{fmt(train.get('max_step'))}</td></tr>
            <tr><th>Loss first</th><td>{fmt(train.get('loss_first'))}</td></tr>
            <tr><th>Loss last</th><td>{fmt(train.get('loss_last'))}</td></tr>
            <tr><th>Eval loss first</th><td>{fmt(train.get('eval_loss_first'))}</td></tr>
            <tr><th>Eval loss last</th><td>{fmt(train.get('eval_loss_last'))}</td></tr>
          </tbody></table>
        </article>
      </div>
    </section>

    {diagnostics_section(diagnostics)}

    <section>
      <h2>Run Notes</h2>
      <div class="note">
        <p>Stage output directory: <code>{esc(summary.get('stage_output_dir'))}</code></p>
        <p>Slurm stdout: <code>{esc(slurm.get('out_path'))}</code></p>
        <p>Slurm stderr: <code>{esc(slurm.get('err_path'))}</code></p>
        <p>Hugging Face remote-code refresh warning observed: <strong>{esc(slurm.get('hf_remote_code_refresh_warning'))}</strong></p>
        <p>Recent progress markers: {esc(slurm.get('completed_samples_markers'))}</p>
      </div>
    </section>
  </main>
</body>
</html>
"""


def render_markdown(summary: dict[str, Any]) -> str:
    candidate = summary["models"].get(STAGE2_KEY)
    active = summary["models"].get(ACTIVE_BASELINE_KEY)
    lines = [
        "# AGVLM SFT Stage Decision",
        "",
        "- Benchmark status: `%s`" % summary["stage_status"],
        "- Decision: **%s**" % summary["decision"],
        "- Stage2 task macro: `%s`" % fmt(metric_get(candidate, "task_macro_average")),
        "- Active SFT task macro: `%s`" % fmt(metric_get(active, "task_macro_average")),
        "",
        "## Promotion Gate",
        "",
        "| Metric | Baseline | Candidate | Preferred Delta | Pass |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["active_gate"].get("checks") or []:
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                row["metric"],
                fmt(row.get("baseline")),
                fmt(row.get("candidate")),
                fmt(row.get("delta_in_preferred_direction")),
                "yes" if row.get("passed") else "no",
            )
        )
    if not summary["active_gate"].get("checks"):
        lines.append("| _No candidate metrics available yet._ |  |  |  | no |")
    lines.extend(["", "## Output Format Diagnostics", ""])
    for key, payload in sorted(summary["output_diagnostics"].items()):
        lines.extend(
            [
                "### %s" % MODEL_LABELS.get(key, key),
                "",
                "- Invalid: `%s` / `%s` (`%s`)"
                % (payload.get("invalid_count"), payload.get("num_examples"), pct(payload.get("invalid_rate"))),
                "- Format-like invalid: `%s` (`%s`)"
                % (payload.get("format_like_invalid_count"), pct(payload.get("format_like_invalid_rate"))),
                "- Format contract issues: `%s` (`%s`)"
                % (payload.get("format_contract_issue_count"), pct(payload.get("format_contract_issue_rate"))),
                "- Out-of-label parseable answers: `%s` (`%s`)"
                % (payload.get("out_of_label_space_count"), pct(payload.get("out_of_label_space_rate"))),
                "- Explicit clarify `Decision:` rate: `%s`" % pct(payload.get("clarify_explicit_decision_rate")),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(args)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "dashboard.html").write_text(render_html(summary), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"dashboard": str(output_dir / "dashboard.html"), "summary": str(output_dir / "summary.json"), "decision": summary["decision"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
