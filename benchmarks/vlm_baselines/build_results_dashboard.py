#!/usr/bin/env python3
"""Build a static HTML dashboard for VLM benchmark metrics and predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
from typing import Any

from utils import BENCHMARK_ROOT, REPO_ROOT, read_jsonl, utc_now, write_json


METRIC_FIELDS = [
    ("task_macro_average", "Task Macro", "Overall task macro average", True),
    ("classification_macro_f1", "Classification F1", "Classification macro F1", True),
    ("vqa_relaxed_accuracy", "VQA Relaxed", "Short VQA relaxed accuracy", True),
    ("clarify_macro_f1", "Clarify F1", "Clarify/respond macro F1", True),
    (
        "consultation_structured_section_compliance",
        "Consultation Sections",
        "Consultation section compliance",
        True,
    ),
    ("invalid_prediction_rate", "Invalid Rate", "Invalid prediction rate", False),
]

TASK_LABELS = {
    "classification": "Classification",
    "vqa": "VQA",
    "clarify_or_respond": "Clarify/Respond",
    "consultation": "Consultation",
}

COLOR_SCALE = ["#2f6fbb", "#b05454", "#2f7d46", "#7a5ea8", "#a86f1d", "#287a6a", "#6b7280"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        action="append",
        default=[],
        help="Benchmark result root. May be provided multiple times. Defaults to benchmarks/vlm_baselines/results.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BENCHMARK_ROOT / "results_dashboard"),
        help="Directory for index.html and dashboard_data.json.",
    )
    parser.add_argument("--title", default="AGVLM Benchmark Results")
    parser.add_argument("--phase", default="", help="Optional phase filter, e.g. sft_benchmark.")
    parser.add_argument("--split", default="", help="Optional split filter, e.g. test.")
    parser.add_argument("--max-examples-per-task-model", type=int, default=1)
    return parser.parse_args()


def _metric_get(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:.3f}"


def _display_model(row: dict[str, Any]) -> str:
    model_key = str(row.get("model_key") or row.get("model_name") or "unknown")
    run_id = str(row.get("run_id") or "")
    return f"{model_key} ({run_id})" if run_id else model_key


def _run_id_for_metrics(path: Path) -> str:
    if path.parent.name == "metrics":
        return path.parent.parent.name
    return path.parent.name


def _resolve_existing_path(value: Any, *, metrics_path: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidates = []
    raw = Path(text).expanduser()
    candidates.append(raw)
    if not raw.is_absolute():
        candidates.append(REPO_ROOT / raw)
        candidates.append(metrics_path.parent.parent / raw)
        candidates.append(metrics_path.parent.parent / "predictions" / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return raw if raw.is_absolute() else REPO_ROOT / raw


def discover_metric_paths(results_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in results_dirs:
        if root.is_file() and root.name.endswith("_metrics.json"):
            candidates = [root]
        elif root.name == "metrics":
            candidates = sorted(root.glob("*_metrics.json"))
        else:
            candidates = sorted(root.glob("**/metrics/*_metrics.json"))
            candidates.extend(sorted(root.glob("*_metrics.json")))
        for candidate in candidates:
            if candidate.name.startswith("summary_table"):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(candidate)
    return sorted(paths)


def metric_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prediction_path = _resolve_existing_path(
        payload.get("prediction_path") or payload.get("predictions_path"),
        metrics_path=path,
    )
    row = {
        "run_id": _run_id_for_metrics(path),
        "phase": payload.get("phase", ""),
        "split": payload.get("split", ""),
        "model_name": payload.get("model_name", ""),
        "model_key": payload.get("model_key", ""),
        "checkpoint_type": payload.get("checkpoint_type", ""),
        "num_examples": payload.get("num_examples", ""),
        "failure_rate": payload.get("failure_rate", ""),
        "invalid_prediction_rate": payload.get("invalid_prediction_rate", ""),
        "task_macro_average": payload.get("task_macro_average", ""),
        "classification_top1_accuracy": _metric_get(payload, "classification.top1_accuracy"),
        "classification_macro_f1": _metric_get(payload, "classification.macro_f1"),
        "vqa_exact_match": _metric_get(payload, "short_vqa.exact_match"),
        "vqa_relaxed_accuracy": _metric_get(payload, "short_vqa.relaxed_accuracy"),
        "vqa_token_f1": _metric_get(payload, "short_vqa.token_f1"),
        "clarify_decision_accuracy": _metric_get(payload, "clarify_or_respond.decision_accuracy"),
        "clarify_macro_f1": _metric_get(payload, "clarify_or_respond.macro_f1"),
        "consultation_structured_section_compliance": _metric_get(
            payload, "consultation.structured_section_compliance"
        ),
        "consultation_management_keyword_coverage": _metric_get(
            payload, "consultation.management_keyword_coverage"
        ),
        "avg_runtime_seconds": payload.get("avg_runtime_seconds", ""),
        "total_runtime_seconds": payload.get("total_runtime_seconds", ""),
        "dtype": payload.get("dtype", ""),
        "quantization": payload.get("quantization", ""),
        "benchmark_manifest_path": payload.get("benchmark_manifest_path", ""),
        "prediction_path": str(prediction_path) if prediction_path else "",
        "metrics_path": str(path),
        "timestamp": payload.get("timestamp", ""),
    }
    return row


def load_metric_rows(results_dirs: list[Path], *, phase: str, split: str) -> list[dict[str, Any]]:
    rows = [metric_row(path) for path in discover_metric_paths(results_dirs)]
    if phase:
        rows = [row for row in rows if row.get("phase") == phase]
    if split:
        rows = [row for row in rows if row.get("split") == split]
    rows.sort(
        key=lambda row: (
            str(row.get("phase") or ""),
            str(row.get("split") or ""),
            str(row.get("run_id") or ""),
            str(row.get("model_key") or row.get("model_name") or ""),
        )
    )
    return rows


def prediction_examples(
    metric_rows: list[dict[str, Any]],
    *,
    max_examples_per_task_model: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for metric in metric_rows:
        path = _resolve_existing_path(metric.get("prediction_path"), metrics_path=Path(metric["metrics_path"]))
        if path is None or not path.is_file():
            continue
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(path):
            task_type = str(row.get("task_type") or "unknown")
            grouped[task_type].append(row)
        for task_type in sorted(grouped):
            rows = grouped[task_type]
            rows.sort(
                key=lambda row: (
                    not bool(row.get("invalid_prediction")),
                    str(row.get("sample_id") or ""),
                )
            )
            for row in rows[:max(0, max_examples_per_task_model)]:
                examples.append(
                    {
                        "run_id": metric.get("run_id", ""),
                        "phase": metric.get("phase", ""),
                        "split": metric.get("split", ""),
                        "model_key": metric.get("model_key", ""),
                        "model_name": metric.get("model_name", ""),
                        "checkpoint_type": metric.get("checkpoint_type", ""),
                        "task_type": task_type,
                        "sample_id": row.get("sample_id", ""),
                        "source_dataset": row.get("source_dataset", ""),
                        "prompt": row.get("prompt", ""),
                        "ground_truth": row.get("ground_truth", ""),
                        "references": row.get("references", []),
                        "raw_output": row.get("raw_output", ""),
                        "parsed_prediction": row.get("parsed_prediction", ""),
                        "parse_status": row.get("parse_status", ""),
                        "invalid_prediction": bool(row.get("invalid_prediction")),
                        "error_message": row.get("error_message"),
                        "image_paths": row.get("image_paths", []),
                    }
                )
    return examples


def _esc(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value or "")
    return html.escape(text, quote=True)


def _value_width(value: Any) -> float:
    number = _as_float(value)
    if number is None:
        return 0.0
    if number <= 1.0:
        return max(0.0, min(100.0, number * 100.0))
    return max(0.0, min(100.0, number))


def chart_html(rows: list[dict[str, Any]], *, field: str, label: str, description: str, color_index: int) -> str:
    sorted_rows = sorted(rows, key=lambda row: (_as_float(row.get(field)) is None, -(_as_float(row.get(field)) or -1)))
    color = COLOR_SCALE[color_index % len(COLOR_SCALE)]
    pieces = [
        '<section class="metric-band">',
        f"<h3>{_esc(label)}</h3>",
        f'<p class="section-note">{_esc(description)}</p>',
        '<div class="bar-chart" role="img" aria-label="%s">' % _esc(label),
    ]
    for row in sorted_rows:
        value = row.get(field)
        width = _value_width(value)
        display = _display_model(row)
        pieces.append(
            '<div class="bar-row">'
            f'<div class="bar-label" title="{_esc(display)}">{_esc(display)}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width: {width:.2f}%; background: {color};"></div>'
            "</div>"
            f'<div class="bar-value">{_esc(_format_number(value))}</div>'
            "</div>"
        )
    pieces.extend(["</div>", "</section>"])
    return "\n".join(pieces)


def metrics_table_html(rows: list[dict[str, Any]]) -> str:
    fields = [
        ("run_id", "Run"),
        ("phase", "Phase"),
        ("split", "Split"),
        ("model_key", "Model"),
        ("checkpoint_type", "Type"),
        ("num_examples", "N"),
        ("task_macro_average", "Task Macro"),
        ("classification_macro_f1", "Class F1"),
        ("vqa_relaxed_accuracy", "VQA Relaxed"),
        ("clarify_macro_f1", "Clarify F1"),
        ("consultation_structured_section_compliance", "Consult Sections"),
        ("invalid_prediction_rate", "Invalid"),
    ]
    header = "".join(f"<th>{_esc(label)}</th>" for _, label in fields)
    body = []
    for row in rows:
        cells = []
        for key, _ in fields:
            value = row.get(key, "")
            if key.endswith("_rate") or key.endswith("_average") or "accuracy" in key or "f1" in key or "compliance" in key:
                value = _format_number(value)
            cells.append(f"<td>{_esc(value)}</td>")
        body.append("<tr>%s</tr>" % "".join(cells))
    return "\n".join(
        [
            '<section class="table-band">',
            "<h2>All Metrics</h2>",
            '<div class="table-wrap">',
            "<table>",
            f"<thead><tr>{header}</tr></thead>",
            "<tbody>%s</tbody>" % "\n".join(body),
            "</table>",
            "</div>",
            "</section>",
        ]
    )


def examples_html(examples: list[dict[str, Any]]) -> str:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_task[str(example.get("task_type") or "unknown")].append(example)
    pieces = ['<section class="examples-band">', "<h2>Prediction Examples</h2>"]
    for task_type in sorted(by_task):
        label = TASK_LABELS.get(task_type, task_type.replace("_", " ").title())
        pieces.append(f'<h3 class="task-heading">{_esc(label)}</h3>')
        for example in by_task[task_type]:
            status_class = "bad" if example.get("invalid_prediction") else "good"
            status = "invalid" if example.get("invalid_prediction") else str(example.get("parse_status") or "parsed")
            title = "%s | %s | %s" % (
                example.get("model_key") or example.get("model_name"),
                example.get("sample_id"),
                status,
            )
            pieces.append(
                '<details class="example-row">'
                f'<summary><span>{_esc(title)}</span><span class="status {status_class}">{_esc(status)}</span></summary>'
                '<div class="example-grid">'
                f'<div><h4>Prompt</h4><pre>{_esc(example.get("prompt"))}</pre></div>'
                f'<div><h4>Ground Truth</h4><pre>{_esc(example.get("ground_truth"))}</pre></div>'
                f'<div><h4>Raw Prediction</h4><pre>{_esc(example.get("raw_output"))}</pre></div>'
                f'<div><h4>Parsed Prediction</h4><pre>{_esc(example.get("parsed_prediction"))}</pre></div>'
                "</div>"
                "</details>"
            )
    pieces.append("</section>")
    return "\n".join(pieces)


def render_html(*, title: str, rows: list[dict[str, Any]], examples: list[dict[str, Any]], data_path: str) -> str:
    generated_at = utc_now()
    model_count = len({(row.get("run_id"), row.get("model_key") or row.get("model_name")) for row in rows})
    phase_count = len({row.get("phase") for row in rows})
    split_count = len({row.get("split") for row in rows})
    charts = "\n".join(
        chart_html(rows, field=field, label=label, description=description, color_index=index)
        for index, (field, label, description, _) in enumerate(METRIC_FIELDS)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --text: #16202a;
      --muted: #5f6b7a;
      --line: #d9e0e8;
      --panel: #ffffff;
      --good: #1f7a4d;
      --bad: #b04444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1, h2, h3, h4, p {{ margin-top: 0; }}
    h1 {{ font-size: 28px; margin-bottom: 12px; letter-spacing: 0; }}
    h2 {{ font-size: 20px; margin-bottom: 18px; letter-spacing: 0; }}
    h3 {{ font-size: 16px; margin-bottom: 6px; letter-spacing: 0; }}
    h4 {{ font-size: 13px; margin-bottom: 8px; color: var(--muted); letter-spacing: 0; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      max-width: 1200px;
    }}
    .summary-item {{
      border-left: 4px solid #2f6fbb;
      padding: 8px 12px;
      background: #f9fbfd;
    }}
    .summary-item strong {{ display: block; font-size: 22px; }}
    .summary-item span {{ color: var(--muted); font-size: 13px; }}
    main {{ padding: 0; }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 0;
      border-bottom: 1px solid var(--line);
    }}
    .metric-band, .table-band, .examples-band {{
      padding: 26px 32px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    .metric-band:nth-child(2n) {{ background: #fbfcfe; }}
    .section-note {{ color: var(--muted); font-size: 13px; margin-bottom: 16px; }}
    .bar-chart {{ display: grid; gap: 9px; }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(160px, 260px) minmax(160px, 1fr) 58px;
      align-items: center;
      gap: 12px;
      min-height: 28px;
    }}
    .bar-label {{
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #243140;
    }}
    .bar-track {{
      height: 13px;
      background: #e7ecf2;
      border-radius: 3px;
      overflow: hidden;
    }}
    .bar-fill {{ height: 100%; border-radius: 3px; }}
    .bar-value {{ font-variant-numeric: tabular-nums; font-size: 12px; color: var(--muted); text-align: right; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 1100px; }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 12px;
      vertical-align: top;
    }}
    th {{ background: #eef3f8; color: #27384a; position: sticky; top: 0; z-index: 1; }}
    .task-heading {{
      margin-top: 24px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
    }}
    .example-row {{
      border: 1px solid var(--line);
      background: #fff;
      margin-bottom: 10px;
    }}
    .example-row summary {{
      cursor: pointer;
      padding: 12px 14px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      font-size: 13px;
    }}
    .status {{
      flex: none;
      border-radius: 3px;
      padding: 2px 7px;
      font-size: 12px;
      color: #fff;
    }}
    .status.good {{ background: var(--good); }}
    .status.bad {{ background: var(--bad); }}
    .example-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 0;
      border-top: 1px solid var(--line);
    }}
    .example-grid > div {{
      padding: 14px;
      border-right: 1px solid var(--line);
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      color: #1f2937;
    }}
    footer {{
      padding: 18px 32px 32px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 720px) {{
      header, .metric-band, .table-band, .examples-band, footer {{ padding-left: 18px; padding-right: 18px; }}
      .chart-grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 4px; }}
      .bar-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{_esc(title)}</h1>
    <div class="summary">
      <div class="summary-item"><strong>{len(rows)}</strong><span>metric rows</span></div>
      <div class="summary-item"><strong>{model_count}</strong><span>model runs</span></div>
      <div class="summary-item"><strong>{phase_count}</strong><span>phases</span></div>
      <div class="summary-item"><strong>{split_count}</strong><span>splits</span></div>
      <div class="summary-item"><strong>{len(examples)}</strong><span>prediction examples</span></div>
    </div>
  </header>
  <main>
    <div class="chart-grid">
      {charts}
    </div>
    {metrics_table_html(rows)}
    {examples_html(examples)}
  </main>
  <footer>
    Generated at {_esc(generated_at)}. Data: {_esc(data_path)}.
  </footer>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    results_dirs = [Path(path) for path in args.results_dir] or [BENCHMARK_ROOT / "results"]
    rows = load_metric_rows(results_dirs, phase=args.phase, split=args.split)
    examples = prediction_examples(
        rows,
        max_examples_per_task_model=args.max_examples_per_task_model,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "title": args.title,
        "generated_at_utc": utc_now(),
        "results_dirs": [str(path) for path in results_dirs],
        "metrics": rows,
        "examples": examples,
    }
    data_path = output_dir / "dashboard_data.json"
    html_path = output_dir / "index.html"
    write_json(data_path, data)
    html_path.write_text(
        render_html(title=args.title, rows=rows, examples=examples, data_path=str(data_path)),
        encoding="utf-8",
    )
    print(json.dumps({"html_path": str(html_path), "data_path": str(data_path), "metrics": len(rows), "examples": len(examples)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
