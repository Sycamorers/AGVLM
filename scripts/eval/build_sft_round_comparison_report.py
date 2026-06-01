#!/usr/bin/env python3
"""Build metrics and side-by-side examples for Phi-4 SFT round comparisons."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "vlm_baselines"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

from dataset_adapter import accepted_references, expected_answer, label_space, system_prompt, user_prompt  # noqa: E402
from metrics import evaluate_prediction_records, parse_prediction_for_metrics  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    checkpoint_type: str
    predictions_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Selected evaluation manifest JSONL.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as key|display_name|checkpoint_type|predictions_jsonl.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--pairwise-md", required=True)
    parser.add_argument("--title", default="SFT Round Inference Comparison")
    parser.add_argument("--phase", default="sft_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--generation-label", default="")
    parser.add_argument("--max-summary-examples", type=int, default=12)
    parser.add_argument("--focus-model-key", default="")
    parser.add_argument("--compare-model-key", default="")
    return parser.parse_args()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_model_spec(raw: str) -> ModelSpec:
    parts = raw.split("|", 3)
    if len(parts) != 4:
        raise ValueError("--model must be formatted as key|display_name|checkpoint_type|predictions_jsonl")
    key, display_name, checkpoint_type, predictions_path = [part.strip() for part in parts]
    if not key:
        raise ValueError("Model key cannot be empty in --model spec.")
    if not display_name:
        display_name = key
    path = Path(predictions_path)
    if not path.is_file():
        raise FileNotFoundError("Missing predictions file for %s: %s" % (key, path))
    return ModelSpec(key=key, display_name=display_name, checkpoint_type=checkpoint_type, predictions_path=path)


def verifier_mode(row: dict[str, Any]) -> str:
    return str((row.get("verifier") or {}).get("mode") or "")


def target_ground_truth(row: dict[str, Any]) -> str:
    target = row.get("target") or {}
    verifier = row.get("verifier") or {}
    if row.get("task_type") == "clarify_or_respond" or verifier.get("mode") == "clarify":
        return str(verifier.get("expected_decision") or target.get("decision") or expected_answer(row))
    if verifier.get("mode") == "label" and target.get("canonical_label"):
        return str(target["canonical_label"])
    return expected_answer(row)


def prediction_map(path: Path, expected_sample_ids: list[str]) -> dict[str, str]:
    expected = set(expected_sample_ids)
    predictions: dict[str, str] = {}
    duplicate_ids: set[str] = set()
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if sample_id in predictions:
            duplicate_ids.add(sample_id)
        value = row.get("prediction")
        if value is None:
            value = row.get("raw_output")
        predictions[sample_id] = str(value or "")
    if duplicate_ids:
        raise ValueError("Duplicate prediction sample ids in %s: %s" % (path, sorted(duplicate_ids)[:10]))
    missing = sorted(expected - set(predictions))
    if missing:
        raise ValueError("Predictions in %s are missing selected sample ids: %s" % (path, missing[:10]))
    return {sample_id: predictions[sample_id] for sample_id in expected_sample_ids}


def records_for_model(
    rows: list[dict[str, Any]],
    predictions_by_sample_id: dict[str, str],
    *,
    spec: ModelSpec,
    labels: list[str],
    phase: str,
    split: str,
    generation_label: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        raw_output = predictions_by_sample_id[sample_id]
        parsed = parse_prediction_for_metrics(
            raw_output=raw_output,
            task_type=str(row.get("task_type") or ""),
            verifier_mode=verifier_mode(row),
            label_space=labels,
        )
        records.append(
            {
                "phase": phase,
                "split": split,
                "model_name": spec.display_name,
                "model_key": spec.key,
                "checkpoint_type": spec.checkpoint_type,
                "sample_id": sample_id,
                "source_dataset": row.get("source_dataset"),
                "task_type": row.get("task_type"),
                "verifier_mode": verifier_mode(row),
                "metadata": row.get("metadata") or {},
                "image_paths": row.get("images") or [],
                "prompt": user_prompt(row),
                "system_prompt": system_prompt(row),
                "ground_truth": target_ground_truth(row),
                "references": accepted_references(row),
                "verifier": row.get("verifier") or {},
                "generation_config": {"label": generation_label},
                "raw_output": raw_output,
                "parsed_prediction": parsed.get("parsed_prediction", ""),
                "normalized_prediction": parsed.get("normalized_prediction", ""),
                "parse_status": parsed.get("parse_status", "missing"),
                "invalid_prediction": bool(parsed.get("invalid_prediction")),
                "sections": parsed.get("sections"),
                "label_mentions": parsed.get("label_mentions"),
                "error_message": None,
            }
        )
    return records


def local_metrics(
    manifest_path: Path,
    ordered_predictions: list[str],
    *,
    metrics_path: Path | None = None,
) -> tuple[dict[str, Any], str | None]:
    if metrics_path is not None and metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8")), None
    try:
        from agri_vlm.data.manifest_io import read_manifest
        from agri_vlm.evaluation.local_eval import score_local_predictions

        typed_rows = read_manifest(manifest_path)
        return score_local_predictions(typed_rows, ordered_predictions), None
    except Exception as exc:  # pragma: no cover - best-effort diagnostic path
        return {}, "%s: %s" % (type(exc).__name__, exc)


def metric_get(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        return "%.6f" % value
    if isinstance(value, int):
        return str(value)
    if value is None:
        return ""
    return str(value)


def md_cell(value: Any, *, max_chars: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        text = "<br>".join(str(item) for item in value)
    else:
        text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    text = text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
    return text if text else "_empty_"


def pairwise_change_counts(sample_ids: list[str], outputs: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    for left, right in itertools.combinations(outputs, 2):
        changed = sum(1 for sample_id in sample_ids if outputs[left][sample_id] != outputs[right][sample_id])
        key = "%s__vs__%s" % (left, right)
        changes[key] = {
            "left": left,
            "right": right,
            "changed": changed,
            "same": len(sample_ids) - changed,
            "changed_rate": changed / float(len(sample_ids)) if sample_ids else 0.0,
        }
    return changes


def reference_for_display(row: dict[str, Any]) -> list[str]:
    refs = accepted_references(row)
    if refs:
        return refs
    expected = expected_answer(row)
    return [expected] if expected else []


def build_comparison_rows(
    rows: list[dict[str, Any]],
    specs: list[ModelSpec],
    outputs: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    comparison_rows = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        payload = {
            "sample_id": sample_id,
            "source_dataset": row.get("source_dataset"),
            "task_type": row.get("task_type"),
            "verifier_mode": verifier_mode(row),
            "images": row.get("images") or [],
            "question_text": user_prompt(row),
            "references": reference_for_display(row),
        }
        for spec in specs:
            payload[spec.key] = outputs[spec.key][sample_id]
        comparison_rows.append(payload)
    return comparison_rows


def choose_summary_examples(
    comparison_rows: list[dict[str, Any]],
    specs: list[ModelSpec],
    *,
    focus_model_key: str,
    compare_model_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if not focus_model_key and len(specs) >= 1:
        focus_model_key = specs[-1].key
    if not compare_model_key and len(specs) >= 2:
        compare_model_key = specs[-2].key
    selected = []
    if focus_model_key and compare_model_key:
        selected = [
            row
            for row in comparison_rows
            if str(row.get(focus_model_key) or "") != str(row.get(compare_model_key) or "")
        ]
    if len(selected) < limit:
        seen = {row["sample_id"] for row in selected}
        selected.extend(row for row in comparison_rows if row["sample_id"] not in seen)
    return selected[:limit]


def metrics_table_lines(specs: list[ModelSpec], metrics_by_model: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Examples | Invalid | Empty | Task Macro | Class Top1 | Class F1 | VQA Relaxed | Clarify F1 | Consultation Structured | Local Avg Reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec in specs:
        metrics = metrics_by_model[spec.key]
        local = metrics.get("local_metrics") or {}
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                md_cell(spec.display_name),
                fmt_metric(metrics.get("num_examples")),
                fmt_metric(metrics.get("num_invalid_predictions")),
                fmt_metric(metrics.get("empty_output_count")),
                fmt_metric(metrics.get("task_macro_average")),
                fmt_metric(metric_get(metrics, "classification.top1_accuracy")),
                fmt_metric(metric_get(metrics, "classification.macro_f1")),
                fmt_metric(metric_get(metrics, "vqa.relaxed_accuracy")),
                fmt_metric(metric_get(metrics, "clarify_or_respond.macro_f1")),
                fmt_metric(metric_get(metrics, "consultation.structured_section_compliance")),
                fmt_metric(local.get("average_reward")),
            )
        )
    return lines


def local_metrics_table_lines(specs: list[ModelSpec], metrics_by_model: dict[str, dict[str, Any]]) -> list[str]:
    metric_names = [
        "label_accuracy",
        "label_macro_f1",
        "answer_exact_match",
        "clarify_accuracy",
        "clarify_precision",
        "clarify_recall",
        "average_reward",
    ]
    lines = [
        "| Metric | %s |" % " | ".join(md_cell(spec.display_name) for spec in specs),
        "| --- | %s |" % " | ".join("---:" for _ in specs),
    ]
    for metric_name in metric_names:
        values = []
        for spec in specs:
            local = metrics_by_model[spec.key].get("local_metrics") or {}
            values.append(fmt_metric(local.get(metric_name)))
        if any(value != "" for value in values):
            lines.append("| %s | %s |" % (metric_name, " | ".join(values)))
    return lines


def pairwise_change_table_lines(specs: list[ModelSpec], changes: dict[str, dict[str, Any]]) -> list[str]:
    display_by_key = {spec.key: spec.display_name for spec in specs}
    lines = [
        "| Pair | Changed | Same | Changed Rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for payload in changes.values():
        pair_name = "%s vs %s" % (
            display_by_key.get(payload["left"], payload["left"]),
            display_by_key.get(payload["right"], payload["right"]),
        )
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                md_cell(pair_name),
                fmt_metric(payload["changed"]),
                fmt_metric(payload["same"]),
                fmt_metric(payload["changed_rate"]),
            )
        )
    return lines


def examples_table_lines(
    rows: list[dict[str, Any]],
    specs: list[ModelSpec],
    *,
    max_chars: int | None,
) -> list[str]:
    header = ["#", "Dataset", "Task", "Sample ID", "Question", "Reference"] + [
        spec.display_name for spec in specs
    ]
    lines = [
        "| %s |" % " | ".join(md_cell(value) for value in header),
        "| %s |" % " | ".join("---:" if index == 0 else "---" for index, _ in enumerate(header)),
    ]
    for index, row in enumerate(rows, start=1):
        values = [
            str(index),
            row.get("source_dataset") or "",
            row.get("task_type") or "",
            "`%s`" % (row.get("sample_id") or ""),
            row.get("question_text") or "",
            row.get("references") or [],
        ]
        values.extend(row.get(spec.key, "") for spec in specs)
        lines.append("| %s |" % " | ".join(md_cell(value, max_chars=max_chars) for value in values))
    return lines


def build_reports(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError("No rows found in manifest: %s" % manifest_path)
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    sample_id_counts = Counter(sample_ids)
    duplicate_sample_ids = sorted(sample_id for sample_id, count in sample_id_counts.items() if count > 1)
    if duplicate_sample_ids:
        raise ValueError("Selected manifest contains duplicate sample ids: %s" % duplicate_sample_ids[:10])

    specs = [parse_model_spec(raw) for raw in args.model]
    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("Model keys must be unique.")

    labels = label_space(rows)
    outputs: dict[str, dict[str, str]] = {}
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    metrics_by_model: dict[str, dict[str, Any]] = {}
    for spec in specs:
        outputs[spec.key] = prediction_map(spec.predictions_path, sample_ids)
        records = records_for_model(
            rows,
            outputs[spec.key],
            spec=spec,
            labels=labels,
            phase=args.phase,
            split=args.split,
            generation_label=args.generation_label,
        )
        records_by_model[spec.key] = records
        metrics = evaluate_prediction_records(records)
        ordered_predictions = [outputs[spec.key][sample_id] for sample_id in sample_ids]
        model_local_metrics, local_error = local_metrics(
            manifest_path,
            ordered_predictions,
            metrics_path=spec.predictions_path.with_name("metrics.json"),
        )
        metrics.update(
            {
                "model_key": spec.key,
                "model_name": spec.display_name,
                "checkpoint_type": spec.checkpoint_type,
                "prediction_path": str(spec.predictions_path),
                "empty_output_count": sum(1 for value in ordered_predictions if not value.strip()),
                "local_metrics": model_local_metrics,
                "local_metrics_error": local_error,
            }
        )
        metrics_by_model[spec.key] = metrics

    changes = pairwise_change_counts(sample_ids, outputs)
    comparison_rows = build_comparison_rows(rows, specs, outputs)
    summary_examples = choose_summary_examples(
        comparison_rows,
        specs,
        focus_model_key=args.focus_model_key,
        compare_model_key=args.compare_model_key,
        limit=args.max_summary_examples,
    )

    output_payload = {
        "manifest": str(manifest_path),
        "num_rows": len(rows),
        "generation_label": args.generation_label,
        "models": {spec.key: metrics_by_model[spec.key] for spec in specs},
        "pairwise_changes": changes,
    }
    write_json(Path(args.output_json), output_payload)
    write_jsonl(Path(args.output_jsonl), comparison_rows)

    summary_lines = [
        "# %s" % args.title,
        "",
        "## Run Summary",
        "",
        "- Manifest: `%s`" % manifest_path,
        "- Rows: `%s`" % len(rows),
        "- Models: `%s`" % ", ".join(spec.display_name for spec in specs),
        "- Generation: `%s`" % (args.generation_label or "see run config"),
        "- Full side-by-side JSONL: `%s`" % args.output_jsonl,
        "- Full pairwise Markdown: `%s`" % args.pairwise_md,
        "",
        "## Benchmark-Style Metrics",
        "",
    ]
    summary_lines.extend(metrics_table_lines(specs, metrics_by_model))
    summary_lines.extend(["", "## Local Reward Metrics", ""])
    summary_lines.extend(local_metrics_table_lines(specs, metrics_by_model))
    summary_lines.extend(["", "## Pairwise Output Changes", ""])
    summary_lines.extend(pairwise_change_table_lines(specs, changes))
    summary_lines.extend(["", "## Inference Examples", ""])
    summary_lines.extend(examples_table_lines(summary_examples, specs, max_chars=700))
    Path(args.summary_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_md).write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")

    pairwise_lines = [
        "# %s Pairwise Examples" % args.title,
        "",
        "## Metrics",
        "",
    ]
    pairwise_lines.extend(metrics_table_lines(specs, metrics_by_model))
    pairwise_lines.extend(["", "## All Predictions", ""])
    pairwise_lines.extend(examples_table_lines(comparison_rows, specs, max_chars=None))
    Path(args.pairwise_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pairwise_md).write_text("\n".join(pairwise_lines).rstrip() + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_md": args.summary_md,
                "pairwise_md": args.pairwise_md,
                "output_json": args.output_json,
                "output_jsonl": args.output_jsonl,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    args = parse_args()
    build_reports(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
