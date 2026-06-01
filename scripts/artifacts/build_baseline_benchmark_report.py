#!/usr/bin/env python3
"""Build a comprehensive external-baseline benchmark report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "vlm_baselines"
sys.path.insert(0, str(BENCHMARK_ROOT))

from utils import load_yaml, model_slug, write_json  # noqa: E402
from model_adapters import MODEL_SPECS  # noqa: E402


PHASE_LABELS = {
    "sft_benchmark": "SFT held-out benchmark",
    "rl_benchmark": "RL held-out benchmark",
}

MODEL_DISPLAY = {
    "huggingfacetb-smolvlm2-2-2b-instruct": "SmolVLM2-2.2B",
    "google-paligemma2-3b-mix-448": "PaliGemma2-3B",
    "microsoft-phi-4-multimodal-instruct": "Phi-4 Multimodal",
    "allenai-molmo2-4b": "Molmo2-4B",
    "llava-hf-llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
    "qwen-qwen2-5-vl-3b-instruct": "Qwen2.5-VL-3B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-output-dir",
        default=str(BENCHMARK_ROOT / "results" / "baseline_report_20260516"),
    )
    parser.add_argument("--report-md", default=str(REPO_ROOT / "reports" / "baseline_benchmark_report.md"))
    parser.add_argument("--metrics-json", default=str(REPO_ROOT / "reports" / "baseline_benchmark_metrics.json"))
    parser.add_argument(
        "--all-metrics-json",
        default=str(REPO_ROOT / "reports" / "baseline_benchmark_all_metrics.json"),
    )
    parser.add_argument(
        "--examples-jsonl",
        default=str(REPO_ROOT / "reports" / "baseline_benchmark_inference_examples.jsonl"),
    )
    parser.add_argument(
        "--examples-md",
        default=str(REPO_ROOT / "reports" / "baseline_benchmark_inference_examples.md"),
    )
    parser.add_argument("--examples-per-task", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def md_cell(value: Any, *, limit: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        value = "%.4f" % value
    elif isinstance(value, (list, tuple)):
        value = "<br>".join(str(item) for item in value)
    else:
        value = str(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if limit is not None and len(value) > limit:
        value = value[: max(limit - 3, 0)].rstrip() + "..."
    value = value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
    return value if value else "_empty_"


def metric_get(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def fmt_rate(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return "%.3f" % float(value)
    return str(value)


def display_model(metrics_or_key: dict[str, Any] | str) -> str:
    if isinstance(metrics_or_key, dict):
        key = str(metrics_or_key.get("model_key") or metrics_or_key.get("model_slug") or "")
        name = str(metrics_or_key.get("model_name") or key)
    else:
        key = str(metrics_or_key)
        name = key
    return MODEL_DISPLAY.get(key) or name.rsplit("/", 1)[-1] or key


def expected_model_keys() -> list[str]:
    payload = load_yaml(BENCHMARK_ROOT / "baseline_models.yaml")
    return [model_slug(str(item.get("name") or item.get("model_name") or "")) for item in payload.get("models", [])]


def adapter_notes() -> list[dict[str, Any]]:
    rows = []
    expected = set(expected_model_keys())
    for model_name, spec in MODEL_SPECS.items():
        model_key = model_slug(model_name)
        if model_key not in expected:
            continue
        rows.append(
            {
                "model_key": model_key,
                "notes": spec.notes,
                "processor_kwargs": spec.processor_kwargs or {},
                "supports_multi_image": spec.supports_multi_image,
                "single_image_policy": spec.single_image_policy,
                "prompt_style": spec.prompt_style,
            }
        )
    return sorted(rows, key=lambda row: display_model(str(row["model_key"])))


def expected_phases() -> list[str]:
    return ["sft_benchmark", "rl_benchmark"]


def load_metrics(output_dir: Path) -> list[dict[str, Any]]:
    metrics = []
    for path in sorted((output_dir / "metrics").glob("*_metrics.json")):
        payload = read_json(path)
        payload["_metrics_path"] = str(path)
        metrics.append(payload)
    metrics.sort(key=lambda item: (str(item.get("phase")), display_model(item)))
    return metrics


def prediction_files(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "predictions").glob("*.jsonl"))


def build_wide_examples(output_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in prediction_files(output_dir):
        for record in read_jsonl(path):
            phase = str(record.get("phase") or "unknown")
            split = str(record.get("split") or "unknown")
            sample_id = str(record.get("sample_id") or "")
            key = (phase, split, sample_id)
            if key not in grouped:
                grouped[key] = {
                    "phase": phase,
                    "split": split,
                    "sample_id": sample_id,
                    "source_dataset": record.get("source_dataset"),
                    "task_type": record.get("task_type"),
                    "verifier_mode": record.get("verifier_mode"),
                    "image_paths": record.get("image_paths") or [],
                    "image_count": record.get("image_count") or len(record.get("image_paths") or []),
                    "prompt": record.get("prompt"),
                    "system_prompt": record.get("system_prompt"),
                    "ground_truth": record.get("ground_truth"),
                    "references": record.get("references") or [],
                    "outputs": {},
                }
            model_key = str(record.get("model_key") or model_slug(str(record.get("model_name") or path.stem)))
            grouped[key]["outputs"][model_key] = {
                "model_name": record.get("model_name"),
                "raw_output": record.get("raw_output"),
                "parsed_prediction": record.get("parsed_prediction"),
                "normalized_prediction": record.get("normalized_prediction"),
                "parse_status": record.get("parse_status"),
                "invalid_prediction": bool(record.get("invalid_prediction")),
                "error_message": record.get("error_message"),
                "runtime_seconds": record.get("runtime_seconds"),
                "image_policy": record.get("image_policy"),
            }
    return [grouped[key] for key in sorted(grouped)]


def split_report() -> dict[str, Any]:
    path = BENCHMARK_ROOT / "splits" / "benchmark_split_report.json"
    return read_json(path) if path.exists() else {}


def best_model(metrics: list[dict[str, Any]], phase: str, dotted_metric: str) -> dict[str, Any] | None:
    candidates = [item for item in metrics if item.get("phase") == phase and metric_get(item, dotted_metric) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(metric_get(item, dotted_metric) or 0.0))


def selected_examples(rows: list[dict[str, Any]], *, examples_per_task: int) -> list[dict[str, Any]]:
    selected = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (str(row.get("phase")), str(row.get("task_type")))
        if counts[key] >= examples_per_task:
            continue
        selected.append(row)
        counts[key] += 1
    return selected


def write_examples_markdown(path: Path, rows: list[dict[str, Any]], model_keys: list[str]) -> None:
    lines = [
        "# Baseline Inference Examples",
        "",
        "This appendix samples rows from the wide JSONL artifact. Raw outputs are truncated here; the JSONL keeps full outputs.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        refs = row.get("references") or [row.get("ground_truth")]
        lines.extend(
            [
                "## Example %s" % index,
                "",
                "- phase: `%s`" % row.get("phase"),
                "- task: `%s`" % row.get("task_type"),
                "- dataset: `%s`" % row.get("source_dataset"),
                "- sample id: `%s`" % row.get("sample_id"),
                "- images: `%s`" % row.get("image_count"),
                "- reference: `%s`" % refs,
                "- prompt: %s" % md_cell(row.get("prompt"), limit=900),
                "",
                "| Model | Parsed | Invalid | Raw output |",
                "| --- | --- | ---: | --- |",
            ]
        )
        outputs = row.get("outputs") or {}
        for model_key in model_keys:
            output = outputs.get(model_key) or {}
            lines.append(
                "| %s | %s | %s | %s |"
                % (
                    md_cell(display_model(model_key)),
                    md_cell(output.get("parsed_prediction"), limit=180),
                    md_cell(output.get("invalid_prediction")),
                    md_cell(output.get("raw_output") or output.get("error_message"), limit=700),
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def phase_model_coverage(metrics: list[dict[str, Any]]) -> dict[str, dict[str, bool]]:
    present = {
        (str(item.get("phase") or ""), str(item.get("model_key") or item.get("model_slug") or ""))
        for item in metrics
    }
    return {
        phase: {model_key: (phase, model_key) in present for model_key in expected_model_keys()}
        for phase in expected_phases()
    }


def missing_phase_models(metrics: list[dict[str, Any]]) -> list[dict[str, str]]:
    coverage = phase_model_coverage(metrics)
    missing = []
    for phase, model_payload in coverage.items():
        for model_key, completed in model_payload.items():
            if not completed:
                missing.append({"phase": phase, "model_key": model_key})
    return missing


def source_dataset_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in metrics:
        for source_dataset, payload in sorted((item.get("per_source_dataset") or {}).items()):
            if not isinstance(payload, dict):
                continue
            rows.append(
                {
                    "phase": item.get("phase"),
                    "model_key": item.get("model_key"),
                    "source_dataset": source_dataset,
                    "num_examples": payload.get("num_examples"),
                    "task_macro_average": payload.get("task_macro_average"),
                    "failure_rate": payload.get("failure_rate"),
                    "invalid_prediction_rate": payload.get("invalid_prediction_rate"),
                    "tasks": payload.get("by_task_type"),
                }
            )
    return rows


def write_all_metrics(path: Path, metrics: list[dict[str, Any]]) -> None:
    write_json(
        path,
        {
            "note": (
                "Combined raw metric payloads. This preserves per-model confusion matrices, "
                "per-class precision/recall/F1, per-source-dataset breakdowns, runtime metadata, "
                "and generation metadata from the evaluator."
            ),
            "metrics": metrics,
        },
    )


def make_summary_payload(
    *,
    output_dir: Path,
    metrics: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    report_md: Path,
    examples_jsonl: Path,
    examples_md: Path,
    all_metrics_json: Path,
) -> dict[str, Any]:
    expected = expected_model_keys()
    completed = sorted({str(item.get("model_key") or "") for item in metrics})
    phases = sorted({str(item.get("phase") or "") for item in metrics})
    by_phase = {
        phase: {
            "best_task_macro": best_model(metrics, phase, "task_macro_average"),
            "best_classification_macro_f1": best_model(metrics, phase, "classification.macro_f1"),
            "best_vqa_relaxed_accuracy": best_model(metrics, phase, "vqa.relaxed_accuracy"),
            "best_clarify_macro_f1": best_model(metrics, phase, "clarify_or_respond.macro_f1"),
            "best_consultation_structured": best_model(metrics, phase, "consultation.structured_section_compliance"),
        }
        for phase in phases
    }
    for phase, payload in by_phase.items():
        for key, value in list(payload.items()):
            if isinstance(value, dict):
                payload[key] = {
                    "model_key": value.get("model_key"),
                    "model_name": value.get("model_name"),
                    "score": metric_get(value, {
                        "best_task_macro": "task_macro_average",
                        "best_classification_macro_f1": "classification.macro_f1",
                        "best_vqa_relaxed_accuracy": "vqa.relaxed_accuracy",
                        "best_clarify_macro_f1": "clarify_or_respond.macro_f1",
                        "best_consultation_structured": "consultation.structured_section_compliance",
                    }[key]),
                }
    return {
        "benchmark_output_dir": str(output_dir),
        "expected_external_baseline_model_keys": expected,
        "completed_model_keys": completed,
        "phases": phases,
        "phase_model_coverage": phase_model_coverage(metrics),
        "missing_phase_models": missing_phase_models(metrics),
        "num_metric_files": len(metrics),
        "num_wide_inference_examples": len(examples),
        "best_by_phase": by_phase,
        "artifacts": {
            "report_md": str(report_md),
            "all_metrics_json": str(all_metrics_json),
            "examples_jsonl": str(examples_jsonl),
            "examples_md": str(examples_md),
            "summary_table": str(output_dir / "metrics" / "summary_table.csv"),
        },
    }


def write_report(path: Path, payload: dict[str, Any], metrics: list[dict[str, Any]], examples: list[dict[str, Any]]) -> None:
    report = split_report()
    model_keys = expected_model_keys()
    lines = [
        "# External Baseline Benchmark Report",
        "",
        "Scope: external baseline VLMs only. Project SFT and RL checkpoints are excluded.",
        "",
        "## Artifacts",
        "",
        "- benchmark output: `%s`" % payload["benchmark_output_dir"],
        "- summary table: `%s`" % payload["artifacts"]["summary_table"],
        "- combined raw metric payloads: `%s`" % payload["artifacts"]["all_metrics_json"],
        "- all wide inference examples: `%s`" % payload["artifacts"]["examples_jsonl"],
        "- sampled example appendix: `%s`" % payload["artifacts"]["examples_md"],
        "",
        "## Coverage",
        "",
        "- metric files: `%s`" % payload["num_metric_files"],
        "- wide inference rows: `%s`" % payload["num_wide_inference_examples"],
        "- expected external baselines: `%s`" % [display_model(key) for key in payload["expected_external_baseline_model_keys"]],
        "- completed model keys: `%s`" % payload["completed_model_keys"],
        "",
        "## Adapter Notes",
        "",
        "| Model | Prompt Style | Multi-Image | Processor Kwargs | Notes |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in adapter_notes():
        multi_image = "yes" if row.get("supports_multi_image") else "no; %s" % row.get("single_image_policy")
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                md_cell(display_model(str(row.get("model_key") or ""))),
                md_cell(row.get("prompt_style")),
                md_cell(multi_image),
                md_cell(row.get("processor_kwargs")),
                md_cell(row.get("notes")),
            )
        )
    lines.extend(
        [
            "",
        "## Completion Matrix",
        "",
        "| Phase | Model | Complete |",
        "| --- | --- | ---: |",
        ]
    )
    for phase, model_payload in payload.get("phase_model_coverage", {}).items():
        for model_key, completed in model_payload.items():
            lines.append("| %s | %s | %s |" % (md_cell(phase), md_cell(display_model(model_key)), "yes" if completed else "no"))
    missing = payload.get("missing_phase_models") or []
    if missing:
        lines.extend(
            [
                "",
                "Missing model/phase metric payloads are listed in `baseline_benchmark_metrics.json`; do not compare rows with incomplete coverage as a full benchmark.",
            ]
        )
    lines.append("")

    phases_payload = report.get("phases") or {}
    if phases_payload:
        lines.extend(
            [
                "## Evaluation Surfaces",
                "",
                "| Phase | Val rows | Test rows | Tasks | Source datasets | Missing images | Train overlap |",
                "| --- | ---: | ---: | --- | --- | ---: | --- |",
            ]
        )
        for phase in ["sft_benchmark", "rl_benchmark"]:
            phase_payload = phases_payload.get(phase) or {}
            rows_by_split = phase_payload.get("rows_by_split") or {}
            overlap = phase_payload.get("train_eval_overlap") or {}
            lines.append(
                "| %s | %s | %s | %s | %s | %s | sample=%s, group=%s |"
                % (
                    md_cell(PHASE_LABELS.get(phase, phase)),
                    rows_by_split.get("val", 0),
                    rows_by_split.get("test", 0),
                    md_cell(phase_payload.get("rows_by_task_type", {})),
                    md_cell(phase_payload.get("rows_by_source_dataset", {})),
                    phase_payload.get("missing_image_sample_count", 0),
                    overlap.get("exact_sample_id_count", 0),
                    overlap.get("group_key_count", 0),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Overall Metrics",
            "",
            "| Phase | Model | N | Fail | Invalid | Task Macro | Avg sec/example |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in metrics:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                md_cell(item.get("phase")),
                md_cell(display_model(item)),
                item.get("num_examples", ""),
                fmt_rate(item.get("failure_rate")),
                fmt_rate(item.get("invalid_prediction_rate")),
                fmt_rate(item.get("task_macro_average")),
                fmt_rate(item.get("avg_runtime_seconds")),
            )
        )
    lines.extend(
        [
            "",
            "## Task Metrics",
            "",
            "| Phase | Model | Class Acc | Class Macro-F1 | VQA Relaxed | VQA Token-F1 | Clarify Acc | Clarify Macro-F1 | Consult Structured | Consult Required | Mgmt Coverage | Forbidden | Overconfident |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in metrics:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                md_cell(item.get("phase")),
                md_cell(display_model(item)),
                fmt_rate(metric_get(item, "classification.top1_accuracy")),
                fmt_rate(metric_get(item, "classification.macro_f1")),
                fmt_rate(metric_get(item, "vqa.relaxed_accuracy")),
                fmt_rate(metric_get(item, "vqa.token_f1")),
                fmt_rate(metric_get(item, "clarify_or_respond.decision_accuracy")),
                fmt_rate(metric_get(item, "clarify_or_respond.macro_f1")),
                fmt_rate(metric_get(item, "consultation.structured_section_compliance")),
                fmt_rate(metric_get(item, "consultation.required_section_compliance")),
                fmt_rate(metric_get(item, "consultation.management_keyword_coverage")),
                fmt_rate(metric_get(item, "consultation.forbidden_claim_rate")),
                fmt_rate(metric_get(item, "consultation.unsafe_or_overconfident_claim_rate")),
            )
            )

    lines.extend(["", "## Best Observed Baselines", ""])
    for phase, bests in payload.get("best_by_phase", {}).items():
        lines.append("- `%s`:" % phase)
        for label, value in bests.items():
            if not value:
                continue
            lines.append(
                "  - %s: `%s` (%s)"
                % (label, display_model(str(value.get("model_key") or "")), fmt_rate(value.get("score")))
            )
    lines.extend(
        [
            "",
            "## Source Dataset Metrics",
            "",
            "| Phase | Model | Source Dataset | N | Task Macro | Fail | Invalid | Tasks |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in source_dataset_rows(metrics):
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                md_cell(row.get("phase")),
                md_cell(display_model(str(row.get("model_key") or ""))),
                md_cell(row.get("source_dataset")),
                row.get("num_examples", ""),
                fmt_rate(row.get("task_macro_average")),
                fmt_rate(row.get("failure_rate")),
                fmt_rate(row.get("invalid_prediction_rate")),
                md_cell(row.get("tasks")),
            )
        )
    lines.extend(
        [
            "",
            "## Fine-Tuning Recommendation",
            "",
            "The external baselines should be treated as weak zero-shot references, not as deployable agriculture consultation systems. Fine-tuning another compact model is justified only if the goal is a second trainable baseline; otherwise prioritize completing and evaluating the current project SFT/RL checkpoints on these same surfaces.",
            "",
            "Candidate selection should be driven by the table above: prefer the smallest model that is competitive on `task_macro_average`, has low failure and invalid rates, and does not depend on brittle remote-code patches. If a second model is trained, use the current benchmark prompts and split manifests unchanged so the comparison remains fair.",
            "",
            "## Metric Limitations",
            "",
            "- Classification labels require exact or unambiguous label-space matches; semantically close common names may still count invalid when outside the canonical label set.",
            "- Consultation metrics are deterministic proxies for structure, keyword coverage, uncertainty, and safety markers; they are not a substitute for agronomic expert review.",
            "- Dense metrics such as per-class precision/recall/F1 and confusion matrices are preserved in the combined raw metrics JSON instead of duplicated into Markdown tables.",
            "- The report is scoped to ground-level RGB agricultural tasks and does not add generic all-purpose VLM behavior.",
            "",
            "## Sampled Inference Examples",
            "",
            "The sampled appendix contains `%s` rows. The JSONL artifact contains all `%s` wide rows with full raw outputs."
            % (min(len(examples), 10**9), payload["num_wide_inference_examples"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.benchmark_output_dir)
    report_md = Path(args.report_md)
    metrics_json = Path(args.metrics_json)
    all_metrics_json = Path(args.all_metrics_json)
    examples_jsonl = Path(args.examples_jsonl)
    examples_md = Path(args.examples_md)

    metrics = load_metrics(output_dir)
    if not metrics:
        raise FileNotFoundError("No metric files found under %s" % (output_dir / "metrics"))
    examples = build_wide_examples(output_dir)
    if not examples:
        raise FileNotFoundError("No prediction files found under %s" % (output_dir / "predictions"))
    model_keys = expected_model_keys()
    sampled = selected_examples(examples, examples_per_task=args.examples_per_task)

    write_jsonl(examples_jsonl, examples)
    write_examples_markdown(examples_md, sampled, model_keys)
    write_all_metrics(all_metrics_json, metrics)
    payload = make_summary_payload(
        output_dir=output_dir,
        metrics=metrics,
        examples=examples,
        report_md=report_md,
        examples_jsonl=examples_jsonl,
        examples_md=examples_md,
        all_metrics_json=all_metrics_json,
    )
    write_json(metrics_json, payload)
    write_report(report_md, payload, metrics, sampled)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
