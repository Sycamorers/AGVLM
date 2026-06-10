#!/usr/bin/env python3
"""Audit SFT data, formatting, training config, and benchmark predictions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "vlm_baselines"))

from metrics import classification_metrics  # noqa: E402
from prediction_parsing import (  # noqa: E402
    detect_ambiguous_label_mentions,
    extract_answer_field,
    extract_label_from_answer,
    normalize_label,
    normalize_text,
)


REPORT_NAMES = [
    "dataset_audit.md",
    "task_distribution.csv",
    "label_distribution.csv",
    "format_audit.md",
    "eval_exact_vs_normalized.md",
    "error_analysis.md",
    "training_config_audit.md",
    "next_round_plan.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        default="data/manifests/full/sft_train_phi4_max3_stage5_closed_label_datafix.jsonl",
    )
    parser.add_argument(
        "--eval-manifest",
        default="data/manifests/full/sft_eval_phi4_max3_stage5_closed_label_stratified1024.jsonl",
    )
    parser.add_argument("--split-dir", default="benchmarks/vlm_baselines/splits_stage5_datafix")
    parser.add_argument("--model-config", default="configs/model/phi4_reasoning_vision_15b_b200.yaml")
    parser.add_argument(
        "--train-config",
        action="append",
        default=[
            "configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage5_datafix.yaml",
            "configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_classification_probe_stage6_mc.yaml",
        ],
    )
    parser.add_argument(
        "--prediction-run",
        action="append",
        default=[
            "stage5=benchmarks/vlm_baselines/results/agvlm_stage5_datafix_benchmark_20260604/predictions/sft-benchmark-agvlm-phi4-sft-stage5-datafix-b200-candidate-test.jsonl",
            "stage6_mc=benchmarks/vlm_baselines/results/agvlm_stage6_mc_benchmark_20260607/predictions/sft-benchmark-agvlm-phi4-sft-classification-probe-stage6-mc-b200-candidate-test.jsonl",
            "stage7_label_only_mixed=benchmarks/vlm_baselines/results/agvlm_stage7_label_only_mixed_benchmark_20260607/predictions/sft-benchmark-agvlm-phi4-sft-stage7-label-only-mixed-b200-candidate-test.jsonl",
            "stage7_label_only_classification=benchmarks/vlm_baselines/results/agvlm_stage7_label_only_classification_benchmark_20260607/predictions/sft-benchmark-agvlm-phi4-sft-stage7-label-only-classification-b200-candidate-test.jsonl",
        ],
        help="Prediction artifact as label=path. May be repeated.",
    )
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--max-error-examples", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSONL in %s line %s" % (path, line_number)) from exc
            if not isinstance(payload, dict):
                raise ValueError("Expected object rows in %s line %s" % (path, line_number))
            rows.append(payload)
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML root must be a mapping: %s" % path)
    return payload


def write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite existing report: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite existing report: %s" % path)
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


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| %s |" % " | ".join(columns),
        "| %s |" % " | ".join("---" for _ in columns),
    ]
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| %s |" % " | ".join(value.replace("\n", " ") for value in values))
    return "\n".join(lines)


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return "%.2f%%" % (100.0 * float(value))


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return ("%." + str(digits) + "f") % float(value)
    except (TypeError, ValueError):
        return str(value)


def message_text(row: dict[str, Any], *, role: str | None = None) -> str:
    chunks: list[str] = []
    for message in row.get("messages") or []:
        if role and message.get("role") != role:
            continue
        content = message.get("content") or []
        if isinstance(content, str):
            chunks.append(content)
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                chunks.append(str(item["text"]))
    return "\n".join(chunks).strip()


def target_text(row: dict[str, Any]) -> str:
    target = row.get("target") or {}
    if target.get("answer_text"):
        return str(target["answer_text"])
    if target.get("canonical_label"):
        return str(target["canonical_label"])
    if target.get("decision"):
        return str(target["decision"])
    if target.get("structured"):
        return json.dumps(target["structured"], sort_keys=True)
    answers = target.get("acceptable_answers") or []
    return str(answers[0]) if answers else ""


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def row_label(row: dict[str, Any]) -> str:
    target = row.get("target") or {}
    verifier = row.get("verifier") or {}
    accepted = verifier.get("accepted_labels") or []
    return str(target.get("canonical_label") or target.get("answer_text") or (accepted[0] if accepted else "")).strip()


def task_name(row: dict[str, Any]) -> str:
    return "%s:%s" % (row.get("source_dataset") or "unknown", row.get("task_type") or "unknown")


def image_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    source = str(row.get("source_dataset") or "")
    image_id = str(metadata.get("source_image_id") or "")
    if not image_id:
        images = row.get("images") or row.get("image_paths") or []
        image_id = str(images[0]) if images else ""
    return "%s::%s" % (source, image_id)


def prompt_target_hash(row: dict[str, Any]) -> str:
    payload = {
        "source_dataset": row.get("source_dataset"),
        "task_type": row.get("task_type"),
        "user": message_text(row, role="user"),
        "target": target_text(row),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def output_format(row: dict[str, Any]) -> str:
    task = str(row.get("task_type") or "")
    target = target_text(row)
    metadata = row.get("metadata") or {}
    if task == "classification":
        if metadata.get("classification_choice_options"):
            return "multiple_choice_choice_answer_evidence"
        if target and "\n" not in target and not re.search(r"(?i)\banswer\s*:", target):
            return "bare_canonical_label_manifest"
        if re.search(r"(?im)^answer\s*:", target) and re.search(r"(?im)^evidence\s*:", target):
            return "answer_evidence"
        return "classification_other"
    if task == "vqa":
        return "short_answer"
    if task == "clarify_or_respond":
        return "decision_plus_answer_or_question"
    if task == "consultation":
        return "structured_sections"
    return "unknown"


def split_rows(train_manifest: Path, eval_manifest: Path, split_dir: Path) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": read_jsonl(train_manifest), "val": [], "test": []}
    val_path = split_dir / "sft_val_manifest.jsonl"
    test_path = split_dir / "sft_test_manifest.jsonl"
    if val_path.exists() and test_path.exists():
        splits["val"] = read_jsonl(val_path)
        splits["test"] = read_jsonl(test_path)
        return splits

    eval_rows = read_jsonl(eval_manifest)
    for row in eval_rows:
        split = str(row.get("split") or "").lower()
        if split in {"validation", "val", "dev"}:
            splits["val"].append(row)
        else:
            splits["test"].append(row)
    return splits


def summarize_tasks(splits: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Counter[str]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    original_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split, rows in splits.items():
        for row in rows:
            key = task_name(row)
            grouped[key][split].append(row)
            if row.get("task_type") == "classification":
                label = normalize_label(row_label(row))
                if label:
                    label_counts["%s/%s" % (key, split)][label] += 1
                    metadata = row.get("metadata") or {}
                    for raw in metadata.get("original_labels") or [metadata.get("original_label")]:
                        if raw:
                            original_labels[(key, label)].add(str(raw))

    task_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    train_label_counts_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for key, by_split in sorted(grouped.items()):
        all_rows = [row for rows in by_split.values() for row in rows]
        sample = all_rows[0]
        task_type = str(sample.get("task_type") or "")
        train_counts = Counter()
        for row in by_split.get("train", []):
            if row.get("task_type") == "classification":
                train_counts[normalize_label(row_label(row))] += 1
        train_label_counts_by_task[key] = train_counts
        class_counts = [count for label, count in train_counts.items() if label]
        output_formats = Counter(output_format(row) for row in all_rows)
        num_classes = len(train_counts) if task_type == "classification" else ""
        min_count = min(class_counts) if class_counts else ""
        max_count = max(class_counts) if class_counts else ""
        risks = major_risks(task_type, by_split, train_counts, output_formats)
        task_rows.append(
            {
                "task_name": key,
                "task_type": task_type,
                "num_train": len(by_split.get("train", [])),
                "num_val": len(by_split.get("val", [])),
                "num_test": len(by_split.get("test", [])),
                "num_classes": num_classes,
                "min_class_count": min_count,
                "max_class_count": max_count,
                "avg_input_words": round(statistics.mean([word_count(message_text(row, role="user")) for row in all_rows]), 2)
                if all_rows
                else 0,
                "avg_output_words": round(statistics.mean([word_count(target_text(row)) for row in all_rows]), 2)
                if all_rows
                else 0,
                "output_format": ";".join("%s=%s" % item for item in sorted(output_formats.items())),
                "major_risk": "; ".join(risks) if risks else "none_observed",
            }
        )
        if task_type == "classification":
            labels = sorted(
                {
                    label
                    for split in ("train", "val", "test")
                    for label in label_counts.get("%s/%s" % (key, split), Counter())
                    if label
                }
            )
            for label in labels:
                label_rows.append(
                    {
                        "task_name": key,
                        "task_type": task_type,
                        "source_dataset": str(sample.get("source_dataset") or ""),
                        "label": label,
                        "train_count": label_counts.get("%s/train" % key, Counter()).get(label, 0),
                        "val_count": label_counts.get("%s/val" % key, Counter()).get(label, 0),
                        "test_count": label_counts.get("%s/test" % key, Counter()).get(label, 0),
                        "original_label_variants": "; ".join(sorted(original_labels.get((key, label), set()))),
                    }
                )
    return task_rows, label_rows, train_label_counts_by_task


def major_risks(
    task_type: str,
    by_split: dict[str, list[dict[str, Any]]],
    train_counts: Counter[str],
    output_formats: Counter[str],
) -> list[str]:
    risks: list[str] = []
    if task_type == "classification":
        if len(train_counts) > 20:
            risks.append("large_label_space")
        if train_counts and min(train_counts.values()) < 20:
            risks.append("low_examples_per_class")
        if train_counts and max(train_counts.values()) / max(min(train_counts.values()), 1) >= 10:
            risks.append("class_skew")
        train_labels = {label for label, count in train_counts.items() if count}
        eval_labels = {
            normalize_label(row_label(row))
            for split in ("val", "test")
            for row in by_split.get(split, [])
            if normalize_label(row_label(row))
        }
        missing = sorted(eval_labels - train_labels)
        if missing:
            risks.append("eval_labels_missing_from_train=%s" % len(missing))
    else:
        risks.append("needs_generation_or_qualitative_metrics")
    if len(output_formats) > 1:
        risks.append("mixed_output_formats")
    if len(by_split.get("train", [])) == 0 and (by_split.get("val") or by_split.get("test")):
        risks.append("no_train_examples_for_eval_task")
    return risks


def leakage_summary(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_sample = {split: {str(row.get("sample_id") or "") for row in rows if row.get("sample_id")} for split, rows in splits.items()}
    by_image = {
        split: {image_key(row) for row in rows if image_key(row) and not image_key(row).endswith("::")}
        for split, rows in splits.items()
    }
    by_hash = {split: {prompt_target_hash(row) for row in rows} for split, rows in splits.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        "sample_id_overlap": {"%s_%s" % pair: len(by_sample[pair[0]] & by_sample[pair[1]]) for pair in pairs},
        "image_group_overlap": {"%s_%s" % pair: len(by_image[pair[0]] & by_image[pair[1]]) for pair in pairs},
        "prompt_target_hash_overlap": {"%s_%s" % pair: len(by_hash[pair[0]] & by_hash[pair[1]]) for pair in pairs},
        "duplicates_within_split": {
            split: len(rows) - len({prompt_target_hash(row) for row in rows}) for split, rows in splits.items()
        },
    }


def label_variant_summary(label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = []
    for row in label_rows:
        raw = [value for value in str(row.get("original_label_variants") or "").split("; ") if value]
        if len({normalize_text(value) for value in raw}) > 1:
            variants.append(
                {
                    "task_name": row["task_name"],
                    "label": row["label"],
                    "variant_count": len(raw),
                    "variants": "; ".join(raw[:8]),
                }
            )
    return sorted(variants, key=lambda item: item["variant_count"], reverse=True)[:20]


def parse_prediction_runs(specs: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--prediction-run must be label=path: %s" % spec)
        label, raw_path = spec.split("=", 1)
        runs.append((label.strip(), ROOT / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)))
    return runs


def accepted_reference_labels(row: dict[str, Any]) -> set[str]:
    refs = {normalize_label(row.get("ground_truth"))}
    for ref in row.get("references") or []:
        refs.add(normalize_label(ref))
    verifier = row.get("verifier") or {}
    for ref in verifier.get("accepted_labels") or []:
        refs.add(normalize_label(ref))
    return {ref for ref in refs if ref}


def label_space_for_prediction(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") or {}
    labels = metadata.get("classification_label_space") or metadata.get("allowed_classification_labels") or []
    if isinstance(labels, list) and labels:
        return [str(label) for label in labels]
    values = list(row.get("references") or [])
    if row.get("ground_truth"):
        values.append(str(row["ground_truth"]))
    verifier = row.get("verifier") or {}
    values.extend(verifier.get("accepted_labels") or [])
    return [str(value) for value in values if str(value).strip()]


def prediction_mode_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        source = str(row.get("source_dataset") or "unknown")
        pred = normalize_label(row.get("normalized_prediction"))
        by_source[source][pred or "<invalid>"] += 1
    rows = []
    for source, counts in sorted(by_source.items()):
        total = sum(counts.values())
        mode, count = counts.most_common(1)[0]
        rows.append({"source_dataset": source, "mode_prediction": mode, "mode_count": count, "total": total, "mode_rate": count / total if total else 0.0})
    return rows


def evaluate_classification_run(
    label: str,
    path: Path,
    train_label_counts_by_task: dict[str, Counter[str]],
    max_error_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    class_rows = [row for row in rows if row.get("task_type") == "classification" or row.get("verifier_mode") == "label"]
    refreshed: list[dict[str, Any]] = []
    raw_exact_correct = 0
    answer_field_exact_correct = 0
    normalized_correct = 0
    mentioned_correct = 0
    ambiguous = 0
    invalid = 0
    normalization_changed = 0
    examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in class_rows:
        label_space = label_space_for_prediction(row)
        parsed = extract_label_from_answer(str(row.get("raw_output") or ""), label_space)
        updated = dict(row)
        updated.update(parsed)
        refreshed.append(updated)
        answer_text, answer_status = extract_answer_field(str(row.get("raw_output") or ""))
        refs = accepted_reference_labels(row)
        parsed_norm = normalize_label(parsed.get("normalized_prediction"))
        raw_exact = bool(row.get("ground_truth")) and str(row.get("raw_output") or "").strip() == str(row.get("ground_truth") or "").strip()
        answer_field_exact = bool(row.get("ground_truth")) and answer_text.strip() == str(row.get("ground_truth") or "").strip()
        norm_ok = parsed_norm in refs
        mention_payload = detect_ambiguous_label_mentions(str(row.get("raw_output") or ""), label_space + list(refs))
        mention_norms = {normalize_label(value) for value in mention_payload.get("matched_labels") or []}
        mentioned = bool(refs & mention_norms) or any(ref and ref in normalize_label(row.get("raw_output")) for ref in refs)
        raw_exact_correct += int(raw_exact)
        answer_field_exact_correct += int(answer_field_exact)
        normalized_correct += int(norm_ok)
        mentioned_correct += int(mentioned)
        ambiguous += int(parsed.get("parse_status") == "ambiguous")
        invalid += int(bool(parsed.get("invalid_prediction")))
        changed = normalize_label(answer_text) != parsed_norm or answer_status not in {"exact", "raw"}
        normalization_changed += int(bool(changed))
        if changed and len(examples) < 30:
            examples.append(
                {
                    "run": label,
                    "sample_id": row.get("sample_id", ""),
                    "reference": row.get("ground_truth", ""),
                    "raw_answer_field": truncate(answer_text, 100),
                    "normalized_prediction": parsed_norm,
                    "parse_status": parsed.get("parse_status", ""),
                    "raw_output": truncate(str(row.get("raw_output") or ""), 180),
                }
            )
        if not norm_ok and len(failures) < max_error_examples:
            failures.append(
                {
                    "run": label,
                    "sample_id": row.get("sample_id", ""),
                    "source_dataset": row.get("source_dataset", ""),
                    "reference": row.get("ground_truth", ""),
                    "prediction": parsed.get("parsed_prediction", ""),
                    "parse_status": parsed.get("parse_status", ""),
                    "error_category": categorize_error(row, parsed, refs, mentioned, train_label_counts_by_task),
                    "raw_output": truncate(str(row.get("raw_output") or ""), 220),
                }
            )

    metrics = classification_metrics(refreshed)
    n = len(class_rows)
    summary = {
        "run": label,
        "prediction_path": str(path),
        "num_classification_examples": n,
        "raw_output_exact_accuracy": raw_exact_correct / n if n else 0.0,
        "answer_field_exact_accuracy": answer_field_exact_correct / n if n else 0.0,
        "normalized_label_accuracy": normalized_correct / n if n else 0.0,
        "label_mentioned_rate": mentioned_correct / n if n else 0.0,
        "ambiguous_prediction_rate": ambiguous / n if n else 0.0,
        "invalid_prediction_rate": invalid / n if n else 0.0,
        "normalization_changed_rate": normalization_changed / n if n else 0.0,
        "macro_f1": metrics.get("macro_f1", 0.0),
        "weighted_f1": metrics.get("weighted_f1", 0.0),
        "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
        "out_of_label_space_rate": metrics.get("out_of_label_space_rate", 0.0),
        "parse_status_counts": metrics.get("parse_status_counts", {}),
        "per_class": metrics.get("per_class", {}),
        "confusion_matrix": metrics.get("confusion_matrix", {}),
        "source_prediction_modes": prediction_mode_rows(refreshed),
    }
    return summary, failures, examples, refreshed


def categorize_error(
    row: dict[str, Any],
    parsed: dict[str, Any],
    refs: set[str],
    label_mentioned: bool,
    train_label_counts_by_task: dict[str, Counter[str]],
) -> str:
    task = "%s:%s" % (row.get("source_dataset") or "unknown", row.get("task_type") or "classification")
    ref = normalize_label(row.get("ground_truth"))
    train_count = train_label_counts_by_task.get(task, Counter()).get(ref, 0)
    status = str(parsed.get("parse_status") or "")
    if status == "ambiguous":
        return "multiple-label ambiguity"
    if label_mentioned:
        return "label mentioned but not selected"
    if status == "out_of_label_space":
        return "synonym/canonical label mismatch or out-of-space output"
    if status == "failed" or parsed.get("invalid_prediction"):
        return "instruction-following failure"
    if train_count and train_count < 20:
        return "insufficient training examples"
    return "true semantic error or source-level prediction collapse"


def truncate(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def confusion_rows(run_label: str, confusion: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    rows = []
    for reference, pred_counts in sorted(confusion.items()):
        for prediction, count in sorted(pred_counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append({"run": run_label, "reference": reference, "prediction": prediction, "count": count})
    return rows


def per_class_metric_rows(run_label: str, per_class: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, metrics in sorted(per_class.items()):
        rows.append(
            {
                "run": run_label,
                "label": label,
                "support": metrics.get("support", 0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
            }
        )
    return rows


def config_value(config: dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(part)
    return default if value is None else value


def trainer_state_summary(config: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for field in ("checkpoint_output_dir", "output_dir"):
        raw = config.get(field)
        if raw:
            run_dir = Path(str(raw))
            candidates.append(run_dir / "trainer_state.json")
            checkpoint_states = sorted(
                run_dir.glob("checkpoint-*/trainer_state.json"),
                key=lambda path: int(re.sub(r"\D+", "", path.parent.name) or "0"),
                reverse=True,
            )
            candidates.extend(checkpoint_states)
    for path in candidates:
        if not path.exists():
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        history = state.get("log_history") or []
        eval_losses = [(item.get("step"), item.get("eval_loss")) for item in history if item.get("eval_loss") is not None]
        train_losses = [(item.get("step"), item.get("loss")) for item in history if item.get("loss") is not None]
        return {
            "trainer_state_path": str(path),
            "best_model_checkpoint": state.get("best_model_checkpoint", ""),
            "best_metric": state.get("best_metric", ""),
            "last_eval_loss": eval_losses[-1][1] if eval_losses else "",
            "last_eval_step": eval_losses[-1][0] if eval_losses else "",
            "last_train_loss": train_losses[-1][1] if train_losses else "",
            "last_train_step": train_losses[-1][0] if train_losses else "",
        }
    return {}


def training_config_rows(model_config_path: Path, train_config_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_config = load_yaml(model_config_path)
    rows = []
    for path in train_config_paths:
        config = load_yaml(path)
        lora = config.get("lora") or {}
        state = trainer_state_summary(config)
        rows.append(
            {
                "config": str(path),
                "base_model": model_config.get("model_name_or_path", ""),
                "manifest_path": config.get("manifest_path", ""),
                "eval_manifest_path": config.get("eval_manifest_path", ""),
                "sft_checkpoint_path": config.get("sft_checkpoint_path", ""),
                "lora_r": lora.get("r", ""),
                "lora_alpha": lora.get("alpha", ""),
                "lora_dropout": lora.get("dropout", ""),
                "target_modules": ",".join(lora.get("target_modules") or []),
                "learning_rate": config.get("learning_rate", ""),
                "per_device_train_batch_size": config.get("per_device_train_batch_size", ""),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps", ""),
                "num_train_epochs": config.get("num_train_epochs", ""),
                "max_steps": config.get("max_steps", ""),
                "warmup_ratio": config.get("warmup_ratio", ""),
                "scheduler": config.get("lr_scheduler_type", "trainer_default"),
                "max_images_per_sample": config.get("max_images_per_sample", ""),
                "precision": "bf16" if config.get("bf16") else ("fp16" if config.get("fp16") else ""),
                "optimizer": config.get("optim", "trainer_default"),
                "deepspeed": config.get("deepspeed", ""),
                "eval_steps": config.get("eval_steps", ""),
                "save_steps": config.get("save_steps", ""),
                "prediction_loss_only": config.get("prediction_loss_only", ""),
                "eval_generation_metrics": config.get("eval_generation_metrics", ""),
                "early_stopping": config.get("early_stopping", "none_configured"),
                **state,
            }
        )
    return rows, model_config


def build_dataset_report(
    splits: dict[str, list[dict[str, Any]]],
    task_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    leakage: dict[str, Any],
    model_config: dict[str, Any],
) -> str:
    total_by_split = {split: len(rows) for split, rows in splits.items()}
    task_type_counts = Counter()
    for split, rows in splits.items():
        for row in rows:
            task_type_counts[(split, row.get("task_type") or "unknown")] += 1
    top_tasks = sorted(task_rows, key=lambda row: int(row.get("num_train") or 0), reverse=True)[:20]
    variants = label_variant_summary(label_rows)
    lines = [
        "# Dataset and Project Scope Audit",
        "",
        "## High-level diagnosis",
        "",
        (
            "The active project is post-training `%s` for ground-level RGB agricultural consultation, "
            "classification, VQA, and clarify/respond behavior with LoRA SFT. The current Stage5 scope is broad: "
            "it mixes closed-label classification, short VQA, structured consultation, and dialogue-routing examples "
            "in one adapter. That is probably too heterogeneous for the available vertical classification signal, "
            "especially because classification has many source-specific label spaces and the benchmark shows source-level "
            "prediction collapse rather than small formatting drift."
            % model_config.get("model_name_or_path", "unknown model")
        ),
        "",
        "## Split sizes",
        "",
        md_table([{"split": split, "num_samples": count} for split, count in total_by_split.items()], ["split", "num_samples"]),
        "",
        "## Task-type mix",
        "",
        md_table(
            [
                {"split": split, "task_type": task, "num_samples": count}
                for (split, task), count in sorted(task_type_counts.items())
            ],
            ["split", "task_type", "num_samples"],
        ),
        "",
        "## Task/domain summary",
        "",
        md_table(
            top_tasks,
            [
                "task_name",
                "task_type",
                "num_train",
                "num_val",
                "num_test",
                "num_classes",
                "min_class_count",
                "max_class_count",
                "output_format",
                "major_risk",
            ],
        ),
        "",
        "Full table: `reports/task_distribution.csv`. Label table: `reports/label_distribution.csv`.",
        "",
        "## Source-label metadata and synonym risks",
        "",
    ]
    if variants:
        lines.append(
            "The table below lists canonical labels that have multiple raw/source label strings attached in metadata. "
            "Some are true synonyms; others may be multi-label metadata or canonicalization collisions requiring manual review."
        )
        lines.append("")
        lines.append(md_table(variants, ["task_name", "label", "variant_count", "variants"]))
    else:
        lines.append("No major original-label variant clusters were detected after canonical normalization.")
    lines.extend(
        [
            "",
            "## Duplicate and leakage checks",
            "",
            "Exact split-overlap counts:",
            "",
            "```json",
            json.dumps(leakage, indent=2, sort_keys=True),
            "```",
            "",
            (
                "Interpretation: sample-id and image-group overlap should be zero for train/test. "
                "Prompt-target hash overlap can be nonzero for repeated generic prompts and repeated labels, "
                "so it is a duplicate-risk signal rather than proof of leakage."
            ),
            "",
            "## Adapter-scope assessment",
            "",
            (
                "The dataset is not a clean fit for one general LoRA adapter if classification accuracy is a primary KPI. "
                "A single adapter may remain useful for agriculture consultation style and short VQA, but classification "
                "should be evaluated and trained as a separate track until label-only behavior and per-source label-space "
                "selection are stable."
            ),
        ]
    )
    return "\n".join(lines)


def build_format_report(task_rows: list[dict[str, Any]], prediction_summaries: list[dict[str, Any]]) -> str:
    rows = [
        {
            "task_name": row["task_name"],
            "task_type": row["task_type"],
            "output_format": row["output_format"],
            "risk": row["major_risk"],
        }
        for row in task_rows
        if row.get("task_type") == "classification" or "mixed_output_formats" in str(row.get("major_risk"))
    ][:40]
    metric_rows = [
        {
            "run": summary["run"],
            "raw_output_exact_accuracy": pct(summary["raw_output_exact_accuracy"]),
            "answer_field_exact_accuracy": pct(summary["answer_field_exact_accuracy"]),
            "normalized_label_accuracy": pct(summary["normalized_label_accuracy"]),
            "normalization_changed_rate": pct(summary["normalization_changed_rate"]),
            "label_mentioned_rate": pct(summary["label_mentioned_rate"]),
        }
        for summary in prediction_summaries
    ]
    return "\n".join(
        [
            "# Input/Output Format Audit",
            "",
            "## Findings",
            "",
            "- Classification manifest targets are stored as bare canonical labels, but instructional SFT renders them as `Answer: <label>` plus `Evidence: ...`.",
            "- Stage6 multiple-choice classification renders `Choice: <letter>`, `Answer: <label>`, and `Evidence: ...`, so it is a different output contract from Stage5 closed-label classification.",
            "- Benchmark classification prompts ask for `Answer` plus optional evidence. Raw-output exact string matching against bare labels is therefore too strict for any output that follows the training format.",
            "- Existing benchmark scoring already uses parser-based normalized labels; the audit adds raw-output exact, extracted-answer exact, normalized-label comparison, and examples where parsing changes the answer.",
            "- The parser now supports line-start Markdown answer fields and JSON `answer`/`label` fields, and still marks multi-label mentions as ambiguous.",
            "",
            "## Classification format by task/source",
            "",
            md_table(rows, ["task_name", "task_type", "output_format", "risk"]),
            "",
            "## Strict vs normalized impact",
            "",
            md_table(metric_rows, ["run", "raw_output_exact_accuracy", "answer_field_exact_accuracy", "normalized_label_accuracy", "normalization_changed_rate", "label_mentioned_rate"]),
            "",
            "## Recommendation",
            "",
            (
                "For classification, standardize the target contract to label-only for classification-specific adapters "
                "or keep `Answer: <label>` but evaluate by an explicit parser. Do not mix `Answer/Evidence` and "
                "`Choice/Answer/Evidence` within the same classification benchmark without tracking them as separate tasks."
            ),
        ]
    )


def build_eval_report(prediction_summaries: list[dict[str, Any]], normalization_examples: list[dict[str, Any]]) -> str:
    rows = []
    for summary in prediction_summaries:
        rows.append(
            {
                "run": summary["run"],
                "n": summary["num_classification_examples"],
                "raw_exact_acc": fmt_float(summary["raw_output_exact_accuracy"]),
                "answer_field_exact_acc": fmt_float(summary["answer_field_exact_accuracy"]),
                "normalized_acc": fmt_float(summary["normalized_label_accuracy"]),
                "macro_f1": fmt_float(summary["macro_f1"]),
                "weighted_f1": fmt_float(summary["weighted_f1"]),
                "balanced_acc": fmt_float(summary["balanced_accuracy"]),
                "ambiguous_rate": fmt_float(summary["ambiguous_prediction_rate"]),
                "invalid_rate": fmt_float(summary["invalid_prediction_rate"]),
                "label_mentioned_rate": fmt_float(summary["label_mentioned_rate"]),
                "oos_rate": fmt_float(summary["out_of_label_space_rate"]),
            }
        )
    mode_rows = []
    for summary in prediction_summaries:
        for row in summary.get("source_prediction_modes") or []:
            mode_rows.append(
                {
                    "run": summary["run"],
                    "source_dataset": row["source_dataset"],
                    "mode_prediction": row["mode_prediction"],
                    "mode_count": row["mode_count"],
                    "total": row["total"],
                    "mode_rate": fmt_float(row["mode_rate"]),
                }
            )
    return "\n".join(
        [
            "# Exact-match vs Normalized Classification Metrics",
            "",
            "## Classification metrics",
            "",
            md_table(
                rows,
                [
                    "run",
                    "n",
                    "raw_exact_acc",
                    "answer_field_exact_acc",
                    "normalized_acc",
                    "macro_f1",
                    "weighted_f1",
                    "balanced_acc",
                    "ambiguous_rate",
                    "invalid_rate",
                    "label_mentioned_rate",
                    "oos_rate",
                ],
            ),
            "",
            "## Source prediction modes",
            "",
            md_table(mode_rows, ["run", "source_dataset", "mode_prediction", "mode_count", "total", "mode_rate"]),
            "",
            "## Normalization examples",
            "",
            md_table(
                normalization_examples[:30],
                ["run", "sample_id", "reference", "raw_answer_field", "normalized_prediction", "parse_status", "raw_output"],
            ),
            "",
            "Confusion matrices are emitted as `reports/confusion_matrix_<run>.csv`.",
            "Per-class precision/recall/F1 tables are emitted as `reports/per_class_metrics_<run>.csv`.",
        ]
    )


def build_error_report(failures: list[dict[str, Any]]) -> str:
    category_counts = Counter(row["error_category"] for row in failures)
    rows = [{"error_category": key, "count_in_sample": value} for key, value in category_counts.most_common()]
    return "\n".join(
        [
            "# Classification Error Analysis",
            "",
            "This report samples failed classification examples after normalized parsing.",
            "",
            "## Failure categories in sampled errors",
            "",
            md_table(rows, ["error_category", "count_in_sample"]),
            "",
            "## Failed examples",
            "",
            md_table(
                failures,
                ["run", "sample_id", "source_dataset", "reference", "prediction", "parse_status", "error_category", "raw_output"],
            ),
            "",
            (
                "The category labels are heuristic and intended to guide manual inspection. "
                "They should not be used to inflate metrics."
            ),
        ]
    )


def build_training_report(config_rows: list[dict[str, Any]]) -> str:
    display = [
        {
            "config": Path(row["config"]).name,
            "base_model": row["base_model"],
            "lora_r": row["lora_r"],
            "alpha": row["lora_alpha"],
            "dropout": row["lora_dropout"],
            "lr": row["learning_rate"],
            "batch": row["per_device_train_batch_size"],
            "grad_accum": row["gradient_accumulation_steps"],
            "max_steps": row["max_steps"],
            "warmup": row["warmup_ratio"],
            "precision": row["precision"],
            "eval_steps": row["eval_steps"],
            "pred_loss_only": row["prediction_loss_only"],
            "gen_metrics": row["eval_generation_metrics"],
            "last_eval_loss": row.get("last_eval_loss", ""),
        }
        for row in config_rows
    ]
    return "\n".join(
        [
            "# Training Configuration Audit",
            "",
            "## Config summary",
            "",
            md_table(
                display,
                [
                    "config",
                    "base_model",
                    "lora_r",
                    "alpha",
                    "dropout",
                    "lr",
                    "batch",
                    "grad_accum",
                    "max_steps",
                    "warmup",
                    "precision",
                    "eval_steps",
                    "pred_loss_only",
                    "gen_metrics",
                    "last_eval_loss",
                ],
            ),
            "",
            "## Assessment",
            "",
            "- LoRA rank 256 / alpha 512 is high-capacity for a heterogeneous SFT adapter and can fit style/format without guaranteeing classification discrimination.",
            "- Stage5 uses loss-only validation (`prediction_loss_only: true`, generation metrics disabled), so decreasing eval loss is not evidence that classification accuracy improved.",
            "- Stage5 starts from an earlier classification-repair adapter but mixes classification, VQA, consultation, and clarify/respond again, creating task-interference risk.",
            "- Stage6 MC is a useful probe but has only 280 train rows and 96 eval rows; it cannot be treated as a complete retraining fix.",
            "- There is no configured early stopping; checkpoint choice should be tied to generation/evaluation metrics, not loss alone.",
        ]
    )


def build_next_plan() -> str:
    return "\n".join(
        [
            "# Recommended Next-round Plan",
            "",
            "## Direct assessment",
            "",
            (
                "The current setup is flawed for diagnosing classification. One mixed LoRA adapter may be acceptable "
                "for broad agriculture assistant behavior, but it is not the right default experiment for fixing many "
                "source-specific, high-cardinality classification tasks. The first fixes should be evaluation and "
                "format standardization, not more blind SFT."
            ),
            "",
            "## Before retraining",
            "",
            "- Freeze the benchmark split and rerun metrics with the robust parser.",
            "- Choose one classification output contract: preferably label-only for classification adapters, or `Answer: <label>` with parser-based scoring.",
            "- Separate classification metrics from VQA, consultation, and clarify/respond metrics.",
            "- Add per-source confusion matrices and source prediction-mode collapse checks to every benchmark.",
            "- Balance low-resource classes and document any synthetic or licensed/manual additions.",
            "",
            "## Data target",
            "",
            (
                "For a real classification repair run, target at least 50-100 clean examples per class for small label spaces, "
                "and 100-300 per class for visually subtle or high-cardinality sources such as IP102. Classes below 20 examples "
                "should be treated as diagnostic-only unless augmented or merged into a scoped label space."
            ),
            "",
            "## Experiments",
            "",
            "A. Evaluation-only fix: keep the current model, apply robust parsing, recompute strict and normalized metrics, and measure metric/output mismatch.",
            "",
            "B. Format-standardized SFT: keep data content, rewrite classification targets to canonical labels only, retrain LoRA, compare exact and normalized metrics.",
            "",
            "C. Task-specific classification LoRA: train only classification-style data with strict label-only output and compare against the mixed Stage5 adapter.",
            "",
            "D. Data scaling test: add or synthesize balanced examples for underrepresented classes, use balanced sampling, and measure low-resource class recall.",
            "",
            "E. General vs specialized adapter comparison: compare one mixed-domain LoRA against source/task-specific adapters with identical benchmark splits.",
            "",
            "## Training configs to try",
            "",
            "- Classification-only adapter: smaller LR sweep around `2e-7`, `5e-7`, `1e-6`; keep deterministic decoding and label-only targets.",
            "- Lower LoRA capacity ablation: compare r=64/128/256 to test whether high-rank mixed SFT is memorizing style/collapse patterns.",
            "- Add eval generation metrics every checkpoint and promote only on normalized accuracy, macro F1, and collapse checks.",
            "- Keep consultation/VQA SFT separate or stage it before classification, then optionally use DPO/GRPO only after SFT format and metrics are stable.",
            "",
            "## Prompt/constrained decoding",
            "",
            (
                "Prompt engineering and constrained decoding can help classification immediately. For closed label spaces, "
                "constrained decoding over labels or option letters should be tested before more training because it directly "
                "addresses invalid/out-of-format outputs without changing weights."
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = split_rows(ROOT / args.train_manifest, ROOT / args.eval_manifest, ROOT / args.split_dir)
    task_rows, label_rows, train_label_counts_by_task = summarize_tasks(splits)
    leakage = leakage_summary(splits)
    config_rows, model_config = training_config_rows(ROOT / args.model_config, [ROOT / path for path in args.train_config])

    prediction_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    normalization_examples: list[dict[str, Any]] = []
    missing_prediction_runs: list[str] = []
    for run_label, path in parse_prediction_runs(args.prediction_run):
        if not path.exists():
            missing_prediction_runs.append("%s=%s" % (run_label, path))
            continue
        summary, run_failures, run_examples, _ = evaluate_classification_run(
            run_label,
            path,
            train_label_counts_by_task,
            args.max_error_examples,
        )
        prediction_summaries.append(summary)
        failures.extend(run_failures)
        normalization_examples.extend(run_examples)
        write_csv(output_dir / ("confusion_matrix_%s.csv" % run_label), confusion_rows(run_label, summary["confusion_matrix"]), overwrite=args.overwrite)
        write_csv(output_dir / ("per_class_metrics_%s.csv" % run_label), per_class_metric_rows(run_label, summary["per_class"]), overwrite=args.overwrite)

    report_paths = {name: output_dir / name for name in REPORT_NAMES}
    write_csv(report_paths["task_distribution.csv"], task_rows, overwrite=args.overwrite)
    write_csv(report_paths["label_distribution.csv"], label_rows, overwrite=args.overwrite)
    write_text(
        report_paths["dataset_audit.md"],
        build_dataset_report(splits, task_rows, label_rows, leakage, model_config),
        overwrite=args.overwrite,
    )
    write_text(report_paths["format_audit.md"], build_format_report(task_rows, prediction_summaries), overwrite=args.overwrite)
    eval_report = build_eval_report(prediction_summaries, normalization_examples)
    if missing_prediction_runs:
        eval_report += "\n\n## Missing prediction artifacts\n\n" + "\n".join("- `%s`" % item for item in missing_prediction_runs)
    write_text(report_paths["eval_exact_vs_normalized.md"], eval_report, overwrite=args.overwrite)
    write_text(report_paths["error_analysis.md"], build_error_report(failures), overwrite=args.overwrite)
    write_text(report_paths["training_config_audit.md"], build_training_report(config_rows), overwrite=args.overwrite)
    write_text(report_paths["next_round_plan.md"], build_next_plan(), overwrite=args.overwrite)

    print("Wrote audit reports:")
    for name in REPORT_NAMES:
        print("  %s" % (output_dir / name))
    for summary in prediction_summaries:
        print(
            "  %s classification: strict=%s normalized=%s macro_f1=%s"
            % (
                summary["run"],
                fmt_float(summary["raw_output_exact_accuracy"]),
                fmt_float(summary["normalized_label_accuracy"]),
                fmt_float(summary["macro_f1"]),
            )
        )


if __name__ == "__main__":
    main()
