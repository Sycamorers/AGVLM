#!/usr/bin/env python3
"""Build a wide pairwise report from Phi4 and benchmark prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "vlm_baselines"
sys.path.insert(0, str(BENCHMARK_ROOT))

from dataset_adapter import accepted_references, expected_answer, label_space, system_prompt, user_prompt  # noqa: E402
from metrics import evaluate_prediction_records, parse_prediction_for_metrics  # noqa: E402
from utils import model_slug, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phi-comparison", required=True)
    parser.add_argument("--benchmark-predictions-dir", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    parser.add_argument("--title", default="Multi-Model Pairwise Inference Report")
    parser.add_argument("--phase", default="rl_benchmark")
    parser.add_argument("--split", default="test")
    parser.add_argument("--generation-label", default="")
    parser.add_argument("--expected-model-name", action="append", default=[])
    parser.add_argument("--failed-model-note", action="append", default=[])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def md_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "<br>".join(str(item) for item in value)
    else:
        value = str(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\\", "\\\\")
    value = value.replace("|", "\\|")
    value = value.replace("\n", "<br>")
    return value if value else "_empty_"


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


def phi_records(
    rows: list[dict[str, Any]],
    predictions_by_sample_id: dict[str, str],
    *,
    model_name: str,
    model_key: str,
    checkpoint_type: str,
    phase: str,
    split: str,
    generation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = label_space(rows)
    records = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        raw_output = predictions_by_sample_id.get(sample_id, "")
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
                "model_name": model_name,
                "model_key": model_key,
                "checkpoint_type": checkpoint_type,
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
                "generation_config": generation_config,
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


def score_model(records: list[dict[str, Any]], *, prediction_path: str) -> dict[str, Any]:
    metrics = evaluate_prediction_records(records)
    metrics.update(
        {
            "model_name": records[0].get("model_name") if records else "",
            "model_key": records[0].get("model_key") if records else "",
            "checkpoint_type": records[0].get("checkpoint_type") if records else "",
            "phase": records[0].get("phase") if records else "",
            "split": records[0].get("split") if records else "",
            "prediction_path": prediction_path,
            "empty_output_count": sum(1 for record in records if not str(record.get("raw_output") or "").strip()),
            "missing_selected_sample_count": sum(1 for record in records if record.get("_missing_selected_sample")),
        }
    )
    return metrics


def display_name(metrics: dict[str, Any]) -> str:
    key = str(metrics.get("model_key") or "").strip()
    if key:
        if key == "phi4_base":
            return "Phi4 Base"
        if key == "phi4_sft":
            return "Phi4 SFT"
        return key
    name = str(metrics.get("model_name") or "model").strip()
    return name.rsplit("/", 1)[-1]


def build_report(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    phi_comparison_path = Path(args.phi_comparison)
    benchmark_predictions_dir = Path(args.benchmark_predictions_dir)
    output_md = Path(args.output_md)
    output_metrics_json = Path(args.output_metrics_json)

    rows = read_jsonl(manifest_path)
    failure_notes = {}
    for note in args.failed_model_note:
        if "=" not in note:
            raise ValueError("--failed-model-note must be formatted as MODEL_NAME=note")
        model_name, message = note.split("=", 1)
        failure_notes[model_slug(model_name)] = message
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    sample_id_set = set(sample_ids)
    row_by_id = {str(row.get("sample_id") or ""): row for row in rows}
    phi_comparison_rows = read_jsonl(phi_comparison_path)
    phi_by_sample_id = {str(row.get("sample_id") or ""): row for row in phi_comparison_rows}
    missing_phi = [sample_id for sample_id in sample_ids if sample_id not in phi_by_sample_id]
    if missing_phi:
        raise ValueError("Phi comparison is missing selected sample ids: %s" % missing_phi[:10])

    generation_config = {
        "source": "phi_comparison",
        "generation_label": args.generation_label,
    }
    base_predictions = {
        sample_id: str(phi_by_sample_id[sample_id].get("base_prediction") or "") for sample_id in sample_ids
    }
    sft_predictions = {
        sample_id: str(phi_by_sample_id[sample_id].get("sft_prediction") or "") for sample_id in sample_ids
    }
    model_outputs: dict[str, dict[str, str]] = {
        "phi4_base": base_predictions,
        "phi4_sft": sft_predictions,
    }
    model_records: dict[str, list[dict[str, Any]]] = {
        "phi4_base": phi_records(
            rows,
            base_predictions,
            model_name="microsoft/Phi-4-reasoning-vision-15b",
            model_key="phi4_base",
            checkpoint_type="base",
            phase=args.phase,
            split=args.split,
            generation_config=generation_config,
        ),
        "phi4_sft": phi_records(
            rows,
            sft_predictions,
            model_name="microsoft/Phi-4-reasoning-vision-15b",
            model_key="phi4_sft",
            checkpoint_type="sft_lora_adapter",
            phase=args.phase,
            split=args.split,
            generation_config=generation_config,
        ),
    }
    prediction_paths_by_key: dict[str, str] = {
        "phi4_base": str(phi_comparison_path),
        "phi4_sft": str(phi_comparison_path),
    }

    prediction_files = sorted(benchmark_predictions_dir.glob("*.jsonl"))
    for prediction_path in prediction_files:
        records = read_jsonl(prediction_path)
        if not records:
            continue
        key = str(records[0].get("model_key") or model_slug(str(records[0].get("model_name") or prediction_path.stem)))
        selected = [record for record in records if str(record.get("sample_id") or "") in sample_id_set]
        by_sample_id = {str(record.get("sample_id") or ""): record for record in selected}
        for sample_id in sample_ids:
            if sample_id not in by_sample_id:
                row = row_by_id[sample_id]
                by_sample_id[sample_id] = {
                    "phase": args.phase,
                    "split": args.split,
                    "model_name": records[0].get("model_name"),
                    "model_key": key,
                    "checkpoint_type": records[0].get("checkpoint_type", "external_baseline"),
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
                    "generation_config": records[0].get("generation_config") or {},
                    "raw_output": "",
                    "parsed_prediction": "",
                    "normalized_prediction": "",
                    "parse_status": "missing",
                    "invalid_prediction": True,
                    "error_message": "Missing selected sample in prediction file.",
                    "_missing_selected_sample": True,
                }
        ordered_records = [by_sample_id[sample_id] for sample_id in sample_ids]
        model_records[key] = ordered_records
        model_outputs[key] = {sample_id: str(by_sample_id[sample_id].get("raw_output") or "") for sample_id in sample_ids}
        prediction_paths_by_key[key] = str(prediction_path)

    for expected_model_name in args.expected_model_name:
        key = model_slug(expected_model_name)
        if key in model_records:
            continue
        missing_records = []
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            missing_records.append(
                {
                    "phase": args.phase,
                    "split": args.split,
                    "model_name": expected_model_name,
                    "model_key": key,
                    "checkpoint_type": "external_baseline",
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
                    "generation_config": {},
                    "raw_output": "",
                    "parsed_prediction": "",
                    "normalized_prediction": "",
                    "parse_status": "missing",
                    "invalid_prediction": True,
                    "error_message": failure_notes.get(
                        key,
                        "No prediction file was produced for this expected benchmark model.",
                    ),
                    "_missing_selected_sample": True,
                }
            )
        model_records[key] = missing_records
        model_outputs[key] = {str(row.get("sample_id") or ""): "" for row in rows}
        prediction_paths_by_key[key] = ""

    metrics_by_model = {
        key: score_model(
            records,
            prediction_path=prediction_paths_by_key.get(key, ""),
        )
        for key, records in model_records.items()
    }

    model_order = ["phi4_base", "phi4_sft"] + sorted(key for key in model_records if key not in {"phi4_base", "phi4_sft"})
    output_payload = {
        "manifest": str(manifest_path),
        "phi_comparison": str(phi_comparison_path),
        "benchmark_predictions_dir": str(benchmark_predictions_dir),
        "num_rows": len(rows),
        "models": {key: metrics_by_model[key] for key in model_order},
    }
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_metrics_json, output_payload)

    lines = [
        "# %s" % args.title,
        "",
        "## Run Summary",
        "",
        "- Manifest: `%s`" % manifest_path,
        "- Rows: `%s`" % len(rows),
        "- Phi4 comparison: `%s`" % phi_comparison_path,
        "- Benchmark predictions: `%s`" % benchmark_predictions_dir,
        "- Generation: `%s`" % (args.generation_label or "see per-model metadata"),
        "- Full prediction cells are not truncated; line breaks render as `<br>`.",
        "",
    ]
    failed_models = [
        (key, metrics_by_model[key])
        for key in model_order
        if metrics_by_model[key].get("num_failed") == metrics_by_model[key].get("num_examples")
        and metrics_by_model[key].get("num_examples")
    ]
    if failed_models:
        lines.extend(
            [
                "## Failed / Unavailable Models",
                "",
            ]
        )
        for key, metrics in failed_models:
            note = failure_notes.get(key) or "No completed predictions were available."
            lines.append("- `%s`: %s" % (display_name(metrics), note))
        lines.append("")
    lines.extend(
        [
        "## Formatting Repair Scope",
        "",
        "- Classification targets use `Answer: <canonical agricultural label>`.",
        "- Clarify/respond targets use explicit `Decision:` plus either `Clarifying question:` or `Answer:`.",
        "- Consultation prompts and targets use line-start `Diagnosis/Evidence/Uncertainty/Management/Follow-up` sections.",
        "",
        "## Metrics",
        "",
        "| Model | Examples | Failed | Invalid | Empty | Task Macro | Classification Top1 | Classification F1 | VQA Relaxed | Clarify Macro F1 | Consultation Structured |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in model_order:
        metrics = metrics_by_model[key]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                md_cell(display_name(metrics)),
                fmt_metric(metrics.get("num_examples")),
                fmt_metric(metrics.get("num_failed")),
                fmt_metric(metrics.get("num_invalid_predictions")),
                fmt_metric(metrics.get("empty_output_count")),
                fmt_metric(metrics.get("task_macro_average")),
                fmt_metric(metric_get(metrics, "classification.top1_accuracy")),
                fmt_metric(metric_get(metrics, "classification.macro_f1")),
                fmt_metric(metric_get(metrics, "vqa.relaxed_accuracy")),
                fmt_metric(metric_get(metrics, "clarify_or_respond.macro_f1")),
                fmt_metric(metric_get(metrics, "consultation.structured_section_compliance")),
            )
        )

    header = [
        "#",
        "Dataset",
        "Task Type",
        "Sample ID",
        "Image(s)",
        "Question / Prompt",
        "Reference",
    ] + [display_name(metrics_by_model[key]) for key in model_order]
    lines.extend(
        [
            "",
            "## Pairwise Predictions",
            "",
            "| %s |" % " | ".join(md_cell(value) for value in header),
            "| %s |" % " | ".join("---:" if index == 0 else "---" for index, _ in enumerate(header)),
        ]
    )
    for index, sample_id in enumerate(sample_ids, start=1):
        row = row_by_id[sample_id]
        refs = accepted_references(row) or [expected_answer(row)]
        values = [
            str(index),
            row.get("source_dataset") or "",
            row.get("task_type") or "",
            "`%s`" % sample_id,
            row.get("images") or [],
            user_prompt(row),
            refs,
        ]
        for key in model_order:
            if sample_id in model_outputs[key]:
                values.append(model_outputs[key][sample_id])
            else:
                values.append("_missing_")
        lines.append("| %s |" % " | ".join(md_cell(value) for value in values))

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(output_md), "output_metrics_json": str(output_metrics_json)}, indent=2))


def main() -> int:
    args = parse_args()
    build_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
