#!/usr/bin/env python3
"""Build the Stage7 label-only classification benchmark report and plots."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "benchmark_report_20260608_stage7_label_only"


RUNS = [
    (
        "Stage2 closed-label",
        ROOT
        / "benchmarks/vlm_baselines/results/agvlm_stage2_b200_retry_full_20260601/metrics/sft-benchmark_agvlm-phi4-sft-classification-repair-instructional-stage2-b200-candidate_test_metrics.json",
    ),
    (
        "Stage3 cls repair",
        ROOT
        / "benchmarks/vlm_baselines/results/agvlm_stage3_closed_label_classification_repair_benchmark_20260601/metrics/sft-benchmark_agvlm-phi4-sft-closed-label-classification-repair-stage3-b200-candidate_test_metrics.json",
    ),
    (
        "Stage4 datafix",
        ROOT
        / "benchmarks/vlm_baselines/results/agvlm_stage4_datafix_benchmark_20260602/metrics/sft-benchmark_agvlm-phi4-sft-stage4-datafix-b200-candidate_test_metrics.json",
    ),
    (
        "Stage5 datafix",
        ROOT
        / "benchmarks/vlm_baselines/results/agvlm_stage5_datafix_benchmark_20260604/metrics/sft-benchmark_agvlm-phi4-sft-stage5-datafix-b200-candidate_test_metrics.json",
    ),
    (
        "Stage6 MC cls-only",
        ROOT
        / "benchmarks/vlm_baselines/results/agvlm_stage6_mc_benchmark_20260607/metrics/sft-benchmark_agvlm-phi4-sft-classification-probe-stage6-mc-b200-candidate_test_metrics.json",
    ),
    (
        "Stage7 label-only cls",
        ROOT
        / "benchmarks/vlm_baselines/results/agvlm_stage7_label_only_classification_benchmark_20260607/metrics/sft-benchmark_agvlm-phi4-sft-stage7-label-only-classification-b200-candidate_test_metrics.json",
    ),
]

STAGE7_PREDICTIONS = (
    ROOT
    / "benchmarks/vlm_baselines/results/agvlm_stage7_label_only_classification_benchmark_20260607/predictions/sft-benchmark-agvlm-phi4-sft-stage7-label-only-classification-b200-candidate-test.jsonl"
)

TRAINING_CSV = (
    ROOT
    / "outputs/artifacts/tables/phi4-reasoning-vision-15b-stage7-label-only-classification-b200-4gpu/training_metrics.csv"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric(payload: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(part)
    return default if value is None else value


def pct(value: Any) -> str:
    if value is None or value == "":
        return ""
    return "%.2f%%" % (100.0 * float(value))


def fnum(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    return ("%." + str(digits) + "f") % float(value)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| %s |" % " | ".join(columns),
        "| %s |" % " | ".join("---" for _ in columns),
    ]
    for row in rows:
        lines.append("| %s |" % " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns))
    return "\n".join(lines)


def load_stage_rows() -> list[dict[str, Any]]:
    rows = []
    for label, path in RUNS:
        payload = read_json(path)
        rows.append(
            {
                "stage": label,
                "num_examples": payload.get("num_examples", ""),
                "task_macro_average": payload.get("task_macro_average"),
                "classification_top1_accuracy": metric(payload, "classification.top1_accuracy"),
                "classification_macro_f1": metric(payload, "classification.macro_f1"),
                "classification_weighted_f1": metric(payload, "classification.weighted_f1"),
                "classification_balanced_accuracy": metric(payload, "classification.balanced_accuracy"),
                "classification_out_of_label_space_rate": metric(payload, "classification.out_of_label_space_rate"),
                "vqa_relaxed_accuracy": metric(payload, "short_vqa.relaxed_accuracy"),
                "vqa_token_f1": metric(payload, "short_vqa.token_f1"),
                "clarify_macro_f1": metric(payload, "clarify_or_respond.macro_f1"),
                "clarify_decision_accuracy": metric(payload, "clarify_or_respond.decision_accuracy"),
                "metrics_path": str(path.relative_to(ROOT)),
            }
        )
    return rows


def stage7_prediction_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    class_rows = [row for row in rows if row.get("task_type") == "classification" or row.get("verifier_mode") == "label"]
    top: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "correct": 0, "out_of_label_space": 0})
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_source_oos: Counter[str] = Counter()
    by_source_correct: Counter[str] = Counter()
    by_source_total: Counter[str] = Counter()
    for row in class_rows:
        pred = str(row.get("normalized_prediction") or "<invalid>")
        source = str(row.get("source_dataset") or "unknown")
        refs = {norm(row.get("ground_truth"))}
        refs.update(norm(ref) for ref in row.get("references") or [])
        verifier = row.get("verifier") or {}
        refs.update(norm(ref) for ref in verifier.get("accepted_labels") or [])
        refs.discard("")
        correct = norm(pred) in refs
        oos = bool(row.get("out_of_label_space"))
        top[pred]["count"] += 1
        top[pred]["correct"] += int(correct)
        top[pred]["out_of_label_space"] += int(oos)
        by_source[source][pred] += 1
        by_source_oos[source] += int(oos)
        by_source_correct[source] += int(correct)
        by_source_total[source] += 1

    top_rows = [
        {
            "prediction": pred,
            "count": stats["count"],
            "correct": stats["correct"],
            "out_of_label_space": stats["out_of_label_space"],
        }
        for pred, stats in sorted(top.items(), key=lambda item: (-item[1]["count"], item[0]))[:20]
    ]
    source_rows = []
    for source, counts in sorted(by_source.items()):
        mode, mode_count = counts.most_common(1)[0]
        total = by_source_total[source]
        source_rows.append(
            {
                "source_dataset": source,
                "total": total,
                "mode_prediction": mode,
                "mode_count": mode_count,
                "mode_rate": mode_count / total if total else 0.0,
                "accuracy": by_source_correct[source] / total if total else 0.0,
                "out_of_label_space_rate": by_source_oos[source] / total if total else 0.0,
            }
        )
    return top_rows, source_rows


def norm(value: Any) -> str:
    return " ".join(str(value or "").lower().strip().split())


def training_rows() -> list[dict[str, Any]]:
    rows = []
    if not TRAINING_CSV.exists():
        return rows
    for row in read_csv(TRAINING_CSV):
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                parsed[key] = None
                continue
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        rows.append(parsed)
    return rows


def save_bar(path: Path, labels: list[str], series: list[tuple[str, list[float]]], ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.8 / max(len(series), 1)
    x = list(range(len(labels)))
    colors = ["#345995", "#03cea4", "#fb4d3d", "#ca7df9", "#f6ae2d"]
    for idx, (name, values) in enumerate(series):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(offsets, values, width=width, label=name, color=colors[idx % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_horizontal_bars(path: Path, rows: list[dict[str, Any]], label_key: str, value_key: str, title: str, xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = list(reversed(rows[:15]))
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [str(row[label_key]) for row in display]
    values = [float(row[value_key]) for row in display]
    ax.barh(labels, values, color="#345995")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_training_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    loss_points = [(row["global_step"], row["loss"]) for row in rows if row.get("loss") is not None]
    eval_points = [(row["global_step"], row["eval_loss"]) for row in rows if row.get("eval_loss") is not None]
    label_acc = [(row["global_step"], row["eval_performance_label_accuracy"]) for row in rows if row.get("eval_performance_label_accuracy") is not None]
    macro_f1 = [(row["global_step"], row["eval_performance_label_macro_f1"]) for row in rows if row.get("eval_performance_label_macro_f1") is not None]
    if loss_points:
        axes[0].plot([x for x, _ in loss_points], [y for _, y in loss_points], color="#345995", label="train loss")
    if eval_points:
        axes[0].scatter([x for x, _ in eval_points], [y for _, y in eval_points], color="#fb4d3d", label="eval loss", zorder=3)
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    if label_acc:
        axes[1].scatter([x for x, _ in label_acc], [y for _, y in label_acc], color="#03cea4", label="eval label accuracy")
    if macro_f1:
        axes[1].scatter([x for x, _ in macro_f1], [y for _, y in macro_f1], color="#ca7df9", label="eval label macro F1")
    axes[1].set_xlabel("global step")
    axes[1].set_ylabel("validation metric")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = REPORT_DIR / "figures"
    tables_dir = REPORT_DIR / "tables"

    stage_rows = load_stage_rows()
    pred_rows = read_jsonl(STAGE7_PREDICTIONS)
    top_rows, source_rows = stage7_prediction_tables(pred_rows)
    train_rows = training_rows()

    write_csv(tables_dir / "stage_progression.csv", stage_rows)
    write_csv(tables_dir / "stage7_top_predictions.csv", top_rows)
    write_csv(tables_dir / "stage7_source_modes.csv", source_rows)

    labels = [row["stage"] for row in stage_rows]
    save_bar(
        figures_dir / "stage_progression_key_metrics.png",
        labels,
        [
            ("classification top1", [float(row["classification_top1_accuracy"] or 0) for row in stage_rows]),
            ("classification macro F1", [float(row["classification_macro_f1"] or 0) for row in stage_rows]),
            ("VQA relaxed", [float(row["vqa_relaxed_accuracy"] or 0) for row in stage_rows]),
            ("clarify macro F1", [float(row["clarify_macro_f1"] or 0) for row in stage_rows]),
        ],
        "score",
    )
    save_bar(
        figures_dir / "classification_oos_rates.png",
        labels,
        [("out-of-label-space", [float(row["classification_out_of_label_space_rate"] or 0) for row in stage_rows])],
        "rate",
    )
    save_bar(
        figures_dir / "stage7_task_scores.png",
        ["task macro", "cls top1", "cls macro F1", "VQA relaxed", "clarify macro F1"],
        [
            (
                "Stage7",
                [
                    float(stage_rows[-1]["task_macro_average"] or 0),
                    float(stage_rows[-1]["classification_top1_accuracy"] or 0),
                    float(stage_rows[-1]["classification_macro_f1"] or 0),
                    float(stage_rows[-1]["vqa_relaxed_accuracy"] or 0),
                    float(stage_rows[-1]["clarify_macro_f1"] or 0),
                ],
            )
        ],
        "score",
    )
    save_horizontal_bars(
        figures_dir / "stage7_prediction_collapse.png",
        top_rows,
        "prediction",
        "count",
        "Stage7 top classification predictions",
        "count",
    )
    save_horizontal_bars(
        figures_dir / "stage7_source_mode_rates.png",
        sorted(source_rows, key=lambda row: float(row["mode_rate"]), reverse=True),
        "source_dataset",
        "mode_rate",
        "Stage7 source-level prediction mode rate",
        "mode rate",
    )
    if train_rows:
        save_training_plot(figures_dir / "stage7_training_curves.png", train_rows)

    stage7 = stage_rows[-1]
    stage5 = stage_rows[3]
    stage6 = stage_rows[4]
    final_logged_loss = next((row for row in reversed(train_rows) if row.get("loss") is not None), {}) if train_rows else {}
    final_train = train_rows[-1] if train_rows else {}
    eval_train_rows = [row for row in train_rows if row.get("eval_loss") is not None or row.get("eval_performance_num_examples") is not None]

    report_rows = []
    for row in stage_rows:
        report_rows.append(
            {
                "stage": row["stage"],
                "examples": row["num_examples"],
                "task macro": pct(row["task_macro_average"]),
                "cls top1": pct(row["classification_top1_accuracy"]),
                "cls macro F1": pct(row["classification_macro_f1"]),
                "cls OOS": pct(row["classification_out_of_label_space_rate"]),
                "VQA relaxed": pct(row["vqa_relaxed_accuracy"]),
                "clarify macro F1": pct(row["clarify_macro_f1"]),
            }
        )

    source_display = [
        {
            "source": row["source_dataset"],
            "n": row["total"],
            "mode": row["mode_prediction"],
            "mode rate": pct(row["mode_rate"]),
            "accuracy": pct(row["accuracy"]),
            "OOS": pct(row["out_of_label_space_rate"]),
        }
        for row in sorted(source_rows, key=lambda item: float(item["total"]), reverse=True)
    ]
    top_display = [
        {
            "prediction": row["prediction"],
            "count": row["count"],
            "correct": row["correct"],
            "OOS": row["out_of_label_space"],
        }
        for row in top_rows[:12]
    ]
    train_eval_summary = {}
    if eval_train_rows:
        train_eval_summary = eval_train_rows[-1]

    report = [
        "# Stage7 Label-only Classification Benchmark",
        "",
        "Date: 2026-06-08",
        "",
        "## Scope",
        "",
        "Stage7 `label_only_classification` was benchmarked on the same Stage5 held-out SFT test split used by the Stage5 and Stage6 MC comparisons.",
        "",
        "- Training job: `34071393`, completed with exit code `0:0`; full-train MaxRSS was `153049588K` under a `256G` request.",
        "- Benchmark job: `34088628`, completed with exit code `0:0`; one GPU, `80G` request, elapsed `00:20:25`.",
        "- Benchmark split: `benchmarks/vlm_baselines/splits_stage5_datafix/sft_test_manifest.jsonl`.",
        "- Prediction file: `benchmarks/vlm_baselines/results/agvlm_stage7_label_only_classification_benchmark_20260607/predictions/sft-benchmark-agvlm-phi4-sft-stage7-label-only-classification-b200-candidate-test.jsonl`.",
        "- Classification benchmark prompt used `AGRI_VLM_CLASSIFICATION_PROMPT_FORMAT=label_only`.",
        "",
        "## Headline",
        "",
        (
            "Stage7 is not promotion-ready. The label-only adapter improved raw formatting for some classification rows, "
            "but did not improve semantic classification. Classification top1 is `%s`, macro F1 is `%s`, and out-of-label-space output is `%s`."
            % (
                pct(stage7["classification_top1_accuracy"]),
                pct(stage7["classification_macro_f1"]),
                pct(stage7["classification_out_of_label_space_rate"]),
            )
        ),
        "",
        (
            "Compared with Stage5, classification top1 changed from `%s` to `%s`, macro F1 from `%s` to `%s`, "
            "and out-of-label-space rate from `%s` to `%s`. This is a regression, not a fix."
            % (
                pct(stage5["classification_top1_accuracy"]),
                pct(stage7["classification_top1_accuracy"]),
                pct(stage5["classification_macro_f1"]),
                pct(stage7["classification_macro_f1"]),
                pct(stage5["classification_out_of_label_space_rate"]),
                pct(stage7["classification_out_of_label_space_rate"]),
            )
        ),
        "",
        "![Stage progression](figures/stage_progression_key_metrics.png)",
        "",
        "![Stage7 task scores](figures/stage7_task_scores.png)",
        "",
        "## Stage Comparison",
        "",
        md_table(report_rows, ["stage", "examples", "task macro", "cls top1", "cls macro F1", "cls OOS", "VQA relaxed", "clarify macro F1"]),
        "",
        "![Classification OOS rates](figures/classification_oos_rates.png)",
        "",
        "## Classification Failure Mode",
        "",
        "The failure is no longer mainly an `Answer:` wrapper mismatch. The label-only prompt causes many classification outputs to be bare strings, but the selected labels are still wrong and often outside the allowed label space.",
        "",
        md_table(top_display, ["prediction", "count", "correct", "OOS"]),
        "",
        "![Stage7 prediction collapse](figures/stage7_prediction_collapse.png)",
        "",
        "## Source-level Modes",
        "",
        md_table(source_display, ["source", "n", "mode", "mode rate", "accuracy", "OOS"]),
        "",
        "![Stage7 source modes](figures/stage7_source_mode_rates.png)",
        "",
        "## Training Curves",
        "",
        (
            "The final logged train loss was `%s` and aggregate train loss was `%s`, but the generated validation metric at step 1000 stayed at label accuracy `%s` and macro F1 `%s` on `%s` examples. "
            "The benchmark result is consistent with that validation signal."
            % (
                fnum(final_logged_loss.get("loss")),
                fnum(final_train.get("train_loss") or final_train.get("loss")),
                fnum(train_eval_summary.get("eval_performance_label_accuracy")),
                fnum(train_eval_summary.get("eval_performance_label_macro_f1")),
                fnum(train_eval_summary.get("eval_performance_num_examples"), digits=0),
            )
        ),
        "",
        "![Stage7 training curves](figures/stage7_training_curves.png)",
        "",
        "## Decision",
        "",
        "- Do not promote Stage7.",
        "- Do not launch another blind LoRA SFT round from this result.",
        "- The next useful experiment is constrained decoding or per-source task-specific adapters with enough balanced examples per class.",
        "- Treat label-only formatting as necessary but insufficient: it reduces wrapper mismatch but does not solve visual discrimination or label-space selection.",
        "",
        "## Generated Artifacts",
        "",
        "- `tables/stage_progression.csv`",
        "- `tables/stage7_top_predictions.csv`",
        "- `tables/stage7_source_modes.csv`",
        "- `figures/*.png`",
        "- Refreshed audit: `reports/eval_exact_vs_normalized.md` and `reports/error_analysis.md`",
    ]
    (REPORT_DIR / "benchmark_report.md").write_text("\n".join(report).rstrip() + "\n", encoding="utf-8")
    (REPORT_DIR / "benchmark_report_summary.json").write_text(
        json.dumps(
            {
                "stage_rows": stage_rows,
                "stage7_top_predictions": top_rows,
                "stage7_source_modes": source_rows,
                "report_path": str((REPORT_DIR / "benchmark_report.md").relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT_DIR / "benchmark_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
