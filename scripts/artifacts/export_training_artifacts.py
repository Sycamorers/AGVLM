#!/usr/bin/env python3
"""Export training metrics into paper-ready tables and curve figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agri_vlm.utils.io import ensure_dir, read_jsonl, write_json


METRIC_GROUPS = {
    "loss": ("loss", "eval_loss", "train_loss"),
    "learning_rate": ("learning_rate",),
    "grad_norm": ("grad_norm",),
    "reward": ("reward", "rewards/", "reward/"),
    "clarify_behavior": ("clarify", "premature_answer", "unnecessary_clarification"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Training run directory under outputs/sft or outputs/rl.")
    parser.add_argument("--metrics-jsonl", default=None, help="Override metrics JSONL path.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="outputs/artifacts")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf"], default=["png", "pdf"])
    return parser.parse_args()


def _resolve_metrics_path(run_dir: Path, explicit: Optional[str]) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([run_dir / "metrics" / "train_metrics.jsonl", run_dir / "metrics.jsonl"])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No metrics JSONL found. Checked: %s" % ", ".join(str(path) for path in candidates)
    )


def _numeric_metric_keys(rows: Iterable[Dict[str, Any]]) -> List[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if key == "global_step":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                keys.add(key)
    return sorted(keys)


def _write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys = ["global_step", *_numeric_metric_keys(rows)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _matching_keys(metric_keys: List[str], needles: Iterable[str]) -> List[str]:
    selected = []
    for key in metric_keys:
        normalized = key.lower()
        if any(needle in normalized for needle in needles):
            selected.append(key)
    return selected


def _plot_group(
    *,
    rows: List[Dict[str, Any]],
    metric_keys: List[str],
    group_name: str,
    needles: Iterable[str],
    output_dir: Path,
    formats: Iterable[str],
) -> List[str]:
    selected = _matching_keys(metric_keys, needles)
    if not selected:
        return []

    import matplotlib.pyplot as plt

    x_values = [row.get("global_step", index) for index, row in enumerate(rows)]
    plotted = False
    plt.figure(figsize=(7.0, 4.0))
    for key in selected:
        y_values = [row.get(key) for row in rows]
        points = [(x, y) for x, y in zip(x_values, y_values) if isinstance(y, (int, float))]
        if not points:
            continue
        plotted = True
        xs, ys = zip(*points)
        plt.plot(xs, ys, label=key, linewidth=1.8)
    if not plotted:
        plt.close()
        return []

    plt.xlabel("Global step")
    plt.ylabel(group_name.replace("_", " ").title())
    plt.title(group_name.replace("_", " ").title())
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    written = []
    ensure_dir(output_dir)
    for fmt in formats:
        path = output_dir / ("%s.%s" % (group_name, fmt))
        plt.savefig(path, dpi=220 if fmt == "png" else None)
        written.append(str(path))
    plt.close()
    return written


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_name = args.run_name or run_dir.name
    metrics_path = _resolve_metrics_path(run_dir, args.metrics_jsonl)
    rows = list(read_jsonl(metrics_path))
    if not rows:
        raise ValueError("Metrics JSONL is empty: %s" % metrics_path)

    output_root = Path(args.output_root)
    figure_dir = output_root / "figures" / run_name
    table_dir = output_root / "tables" / run_name
    report_dir = output_root / "reports" / run_name
    metrics_csv = table_dir / "training_metrics.csv"
    _write_metrics_csv(metrics_csv, rows)

    metric_keys = _numeric_metric_keys(rows)
    figures: Dict[str, List[str]] = {}
    missing_groups = []
    not_plotted_groups = []
    plotting_error = None
    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception as exc:
        plotting_error = "matplotlib unavailable: %s" % exc

    if plotting_error:
        for group_name, needles in METRIC_GROUPS.items():
            if _matching_keys(metric_keys, needles):
                not_plotted_groups.append(group_name)
            else:
                missing_groups.append(group_name)
    else:
        for group_name, needles in METRIC_GROUPS.items():
            written = _plot_group(
                rows=rows,
                metric_keys=metric_keys,
                group_name=group_name,
                needles=needles,
                output_dir=figure_dir,
                formats=args.formats,
            )
            if written:
                figures[group_name] = written
            else:
                if _matching_keys(metric_keys, needles):
                    not_plotted_groups.append(group_name)
                else:
                    missing_groups.append(group_name)

    manifest = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "metrics_jsonl": str(metrics_path),
        "metrics_csv": str(metrics_csv),
        "figures": figures,
        "missing_metric_groups": missing_groups,
        "not_plotted_groups": not_plotted_groups,
        "plotting_error": plotting_error,
        "available_numeric_metrics": metric_keys,
    }
    manifest_path = report_dir / "training_artifact_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
