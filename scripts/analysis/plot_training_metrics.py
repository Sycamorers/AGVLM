#!/usr/bin/env python3
"""Plot SFT training metrics saved as JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-jsonl", required=True, help="Path to a training metrics.jsonl file.")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and summary files.")
    parser.add_argument("--title", default="", help="Human-readable run title.")
    parser.add_argument("--rolling-window", type=int, default=100, help="Rolling window for smoothed curves.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON in %s line %s" % (path, line_number)) from exc
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object in %s line %s" % (path, line_number))
            rows.append(payload)
    if not rows:
        raise ValueError("No metrics rows found in %s" % path)
    return rows


def numeric_rows(rows: Sequence[Dict[str, Any]], field: str) -> List[Dict[str, float]]:
    output = []
    for row in rows:
        if field not in row or "global_step" not in row:
            continue
        try:
            value = float(row[field])
            step = float(row["global_step"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or not math.isfinite(step):
            continue
        payload = {"global_step": step, field: value}
        if "epoch" in row:
            try:
                payload["epoch"] = float(row["epoch"])
            except (TypeError, ValueError):
                pass
        output.append(payload)
    return output


def rolling(values: Sequence[float], window: int) -> List[float]:
    if not values:
        return []
    window = max(1, min(window, len(values)))
    smoothed = []
    total = 0.0
    queue: List[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        smoothed.append(total / len(queue))
    return smoothed


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    return plt


def _series(rows: Sequence[Dict[str, float]], field: str) -> tuple[List[float], List[float]]:
    return [row["global_step"] for row in rows], [row[field] for row in rows]


def _save_line_plot(
    *,
    plt,
    rows: Sequence[Dict[str, float]],
    field: str,
    path: Path,
    title: str,
    ylabel: str,
    rolling_window: int,
    raw_alpha: float = 0.25,
) -> None:
    if not rows:
        return
    steps, values = _series(rows, field)
    smoothed = rolling(values, rolling_window)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, values, color="#8aa0b5", alpha=raw_alpha, linewidth=0.8, label="raw")
    ax.plot(steps, smoothed, color="#1f77b4", linewidth=1.8, label="rolling mean (%s)" % min(rolling_window, len(values)))
    min_index = min(range(len(values)), key=lambda index: values[index])
    ax.scatter([steps[min_index]], [values[min_index]], color="#d62728", s=24, zorder=3, label="min %.4f" % values[min_index])
    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_eval_plot(*, plt, rows: Sequence[Dict[str, float]], path: Path, title: str) -> None:
    if not rows:
        return
    steps, values = _series(rows, "eval_loss")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, values, marker="o", color="#9467bd", linewidth=1.6)
    for step, value in zip(steps, values):
        ax.annotate("%.3f" % value, (step, value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel("eval loss")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_histogram(*, plt, rows: Sequence[Dict[str, float]], field: str, path: Path, title: str) -> None:
    if not rows:
        return
    _steps, values = _series(rows, field)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=min(60, max(10, int(math.sqrt(len(values))))), color="#2ca02c", alpha=0.78)
    ax.axvline(mean(values), color="#d62728", linewidth=1.5, label="mean %.3f" % mean(values))
    ax.axvline(median(values), color="#ff7f0e", linewidth=1.5, label="median %.3f" % median(values))
    ax.set_title(title)
    ax.set_xlabel(field)
    ax.set_ylabel("count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_overview(
    *,
    plt,
    loss_rows: Sequence[Dict[str, float]],
    grad_rows: Sequence[Dict[str, float]],
    lr_rows: Sequence[Dict[str, float]],
    eval_rows: Sequence[Dict[str, float]],
    path: Path,
    title: str,
    rolling_window: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    if loss_rows:
        steps, values = _series(loss_rows, "loss")
        axes[0].plot(steps, values, color="#8aa0b5", alpha=0.22, linewidth=0.7)
        axes[0].plot(steps, rolling(values, rolling_window), color="#1f77b4", linewidth=1.6)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("global step")
        axes[0].set_ylabel("loss")
    if grad_rows:
        steps, values = _series(grad_rows, "grad_norm")
        axes[1].plot(steps, values, color="#c7a76c", alpha=0.35, linewidth=0.7)
        axes[1].plot(steps, rolling(values, rolling_window), color="#ff7f0e", linewidth=1.6)
        axes[1].set_title("Gradient Norm")
        axes[1].set_xlabel("global step")
        axes[1].set_ylabel("grad norm")
    if lr_rows:
        steps, values = _series(lr_rows, "learning_rate")
        axes[2].plot(steps, values, color="#2ca02c", linewidth=1.6)
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("global step")
        axes[2].set_ylabel("learning rate")
    if eval_rows:
        steps, values = _series(eval_rows, "eval_loss")
        axes[3].plot(steps, values, marker="o", color="#9467bd", linewidth=1.5)
        axes[3].set_title("Eval Loss")
        axes[3].set_xlabel("global step")
        axes[3].set_ylabel("eval loss")
    else:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "No eval_loss rows", ha="center", va="center")
    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(rows: Sequence[Dict[str, Any]], path: Path, title: str) -> Dict[str, Any]:
    loss_rows = numeric_rows(rows, "loss")
    eval_rows = numeric_rows(rows, "eval_loss")
    grad_rows = numeric_rows(rows, "grad_norm")
    lr_rows = numeric_rows(rows, "learning_rate")
    final_train = next((row for row in reversed(rows) if "train_loss" in row), {})

    def stats(series_rows: Sequence[Dict[str, float]], field: str) -> Dict[str, Any]:
        if not series_rows:
            return {}
        values = [row[field] for row in series_rows]
        return {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "median": median(values),
        }

    summary = {
        "title": title,
        "metrics_jsonl": str(path),
        "rows": len(rows),
        "loss": stats(loss_rows, "loss"),
        "eval_loss": stats(eval_rows, "eval_loss"),
        "grad_norm": stats(grad_rows, "grad_norm"),
        "learning_rate": stats(lr_rows, "learning_rate"),
        "first_step": int(loss_rows[0]["global_step"]) if loss_rows else None,
        "last_step": int(loss_rows[-1]["global_step"]) if loss_rows else None,
        "first_epoch": loss_rows[0].get("epoch") if loss_rows else None,
        "last_epoch": loss_rows[-1].get("epoch") if loss_rows else None,
        "final_train_summary": final_train,
    }
    if loss_rows:
        values = [row["loss"] for row in loss_rows]
        early_window = values[: min(100, len(values))]
        late_window = values[-min(100, len(values)) :]
        summary["loss"]["first_100_mean"] = mean(early_window)
        summary["loss"]["last_100_mean"] = mean(late_window)
        summary["loss"]["first_to_last_100_delta"] = mean(late_window) - mean(early_window)
    return summary


def write_summary_markdown(output_path: Path, summary: Dict[str, Any], plots: Sequence[str]) -> None:
    lines = [
        "# Training Metrics Graphs",
        "",
        "## Run",
        "",
        "- Title: `%s`" % summary["title"],
        "- Metrics: `%s`" % summary["metrics_jsonl"],
        "- Rows: `%s`" % summary["rows"],
        "- Step range: `%s` to `%s`" % (summary["first_step"], summary["last_step"]),
        "- Epoch range: `%s` to `%s`" % (summary["first_epoch"], summary["last_epoch"]),
        "",
        "## Key Numbers",
        "",
        "| Metric | Count | First | Last | Min | Max | Mean | Median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field in ["loss", "eval_loss", "grad_norm", "learning_rate"]:
        stats = summary.get(field) or {}
        if not stats:
            continue
        lines.append(
            "| %s | %s | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g |"
            % (
                field,
                stats["count"],
                stats["first"],
                stats["last"],
                stats["min"],
                stats["max"],
                stats["mean"],
                stats["median"],
            )
        )
    if summary.get("loss"):
        loss_stats = summary["loss"]
        lines.extend(
            [
                "",
                "## Loss Trend",
                "",
                "- First 100-step mean: `%.6f`" % loss_stats["first_100_mean"],
                "- Last 100-step mean: `%.6f`" % loss_stats["last_100_mean"],
                "- Delta: `%.6f`" % loss_stats["first_to_last_100_delta"],
            ]
        )
    if summary.get("final_train_summary"):
        lines.extend(["", "## Final Train Summary", ""])
        for key, value in sorted(summary["final_train_summary"].items()):
            lines.append("- `%s`: `%s`" % (key, value))
    lines.extend(["", "## Plots", ""])
    for plot in plots:
        lines.append("- `%s`" % plot)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or metrics_path.parent.name
    rows = read_jsonl(metrics_path)
    loss_rows = numeric_rows(rows, "loss")
    eval_rows = numeric_rows(rows, "eval_loss")
    grad_rows = numeric_rows(rows, "grad_norm")
    lr_rows = numeric_rows(rows, "learning_rate")

    plt = _setup_matplotlib()
    plots = []
    plot_specs = [
        ("overview.png", lambda p: _save_overview(plt=plt, loss_rows=loss_rows, grad_rows=grad_rows, lr_rows=lr_rows, eval_rows=eval_rows, path=p, title=title, rolling_window=args.rolling_window)),
        ("loss_curve.png", lambda p: _save_line_plot(plt=plt, rows=loss_rows, field="loss", path=p, title="%s: Training Loss" % title, ylabel="loss", rolling_window=args.rolling_window)),
        ("grad_norm_curve.png", lambda p: _save_line_plot(plt=plt, rows=grad_rows, field="grad_norm", path=p, title="%s: Gradient Norm" % title, ylabel="grad norm", rolling_window=args.rolling_window, raw_alpha=0.35)),
        ("learning_rate_curve.png", lambda p: _save_line_plot(plt=plt, rows=lr_rows, field="learning_rate", path=p, title="%s: Learning Rate" % title, ylabel="learning rate", rolling_window=1, raw_alpha=0.0)),
        ("loss_histogram.png", lambda p: _save_histogram(plt=plt, rows=loss_rows, field="loss", path=p, title="%s: Loss Distribution" % title)),
        ("eval_loss_curve.png", lambda p: _save_eval_plot(plt=plt, rows=eval_rows, path=p, title="%s: Eval Loss" % title)),
    ]
    for filename, writer in plot_specs:
        path = output_dir / filename
        before_exists = path.exists()
        writer(path)
        if path.exists() and (not before_exists or path.stat().st_size > 0):
            plots.append(filename)

    summary = summarize(rows, metrics_path, title)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_markdown(output_dir / "summary.md", summary, plots)
    print("Wrote %s plots to %s" % (len(plots), output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
