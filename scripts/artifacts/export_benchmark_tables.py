#!/usr/bin/env python3
"""Export benchmark summaries into reusable result tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.utils.io import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("MODEL_NAME", "SUMMARY_OR_DIR"),
        required=True,
        help="Model label plus benchmark summary.json path or its parent directory.",
    )
    parser.add_argument("--output-root", default="outputs/artifacts")
    parser.add_argument("--table-name", default="benchmark_results")
    return parser.parse_args()


def _summary_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_dir():
        path = path / "summary.json"
    if not path.exists():
        raise FileNotFoundError("Benchmark summary not found: %s" % path)
    return path


def _flatten_summary(model_name: str, summary_path: Path) -> List[Dict[str, Any]]:
    summary = read_json(summary_path)
    rows = []
    for task in summary.get("tasks", []):
        row: Dict[str, Any] = {
            "model_name": model_name,
            "task": task.get("task"),
            "checkpoint_path": summary.get("checkpoint_path"),
            "prediction_mode": summary.get("prediction_mode"),
            "manifest_path": task.get("manifest_path"),
            "metrics_path": task.get("metrics_path"),
            "predictions_path": task.get("predictions_path"),
            "num_predictions": task.get("num_predictions"),
        }
        for key, value in sorted((task.get("metrics") or {}).items()):
            row["metric_%s" % key] = value
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys = sorted({key for row in rows for key in row})
    preferred = [
        "model_name",
        "task",
        "checkpoint_path",
        "prediction_mode",
        "num_predictions",
        "manifest_path",
        "metrics_path",
        "predictions_path",
    ]
    fieldnames = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    metric_keys = sorted({key for row in rows for key in row if key.startswith("metric_")})
    columns = ["model_name", "task", "num_predictions", *metric_keys]
    lines = [
        "# Benchmark Results",
        "",
        "| %s |" % " | ".join(columns),
        "| %s |" % " | ".join("---" for _ in columns),
    ]
    for row in rows:
        lines.append("| %s |" % " | ".join(str(row.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    summaries = []
    for model_name, path_like in args.run:
        summary_path = _summary_path(path_like)
        summaries.append({"model_name": model_name, "summary_path": str(summary_path)})
        rows.extend(_flatten_summary(model_name, summary_path))
    if not rows:
        raise ValueError("No benchmark task rows were found in the provided summaries.")

    output_root = Path(args.output_root)
    csv_path = output_root / "tables" / ("%s.csv" % args.table_name)
    markdown_path = output_root / "tables" / ("%s.md" % args.table_name)
    manifest_path = output_root / "reports" / ("%s_manifest.json" % args.table_name)
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows)
    manifest = {
        "rows": len(rows),
        "summaries": summaries,
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
