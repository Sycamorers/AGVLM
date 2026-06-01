#!/usr/bin/env python3
"""Audit rendered SFT targets for degenerate or weak supervision patterns."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from statistics import mean, median
from typing import Any

from agri_vlm.data.conversation_format import INSTRUCTIONAL_FORMAT, sample_to_prompt_messages, target_to_text
from agri_vlm.data.manifest_io import read_manifest
from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.io import ensure_dir, write_json


GENERIC_ANSWER_TERMS = {
    "answer",
    "aphid",
    "crop",
    "crops",
    "disease",
    "insect",
    "leaf",
    "leaves",
    "pest",
    "plant",
    "plants",
    "symptom",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--target-format", default=INSTRUCTIONAL_FORMAT)
    parser.add_argument("--short-answer-token-threshold", type=int, default=2)
    parser.add_argument("--max-examples-per-flag", type=int, default=20)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def answer_after_prefix(rendered_target: str) -> str:
    text = rendered_target.strip()
    if text.lower().startswith("answer:"):
        return text.split(":", 1)[1].strip()
    return text


def token_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", value))


def first_user_text(sample: UnifiedSample) -> str:
    messages = sample_to_prompt_messages(sample, prompt_format=INSTRUCTIONAL_FORMAT)
    for message in messages:
        if message.get("role") != "user":
            continue
        for content in message.get("content") or []:
            if content.get("type") == "text":
                return str(content.get("text") or "").strip()
    return ""


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count)


def add_example(
    examples_by_flag: dict[str, list[dict[str, Any]]],
    flag: str,
    sample: UnifiedSample,
    rendered_target: str,
    *,
    limit: int,
) -> None:
    if len(examples_by_flag[flag]) >= limit:
        return
    examples_by_flag[flag].append(
        {
            "sample_id": sample.sample_id,
            "source_dataset": sample.source_dataset,
            "task_type": sample.task_type,
            "verifier_mode": sample.verifier.mode,
            "images": list(sample.images),
            "prompt": first_user_text(sample),
            "target": rendered_target,
            "target_answer": answer_after_prefix(rendered_target),
            "canonical_label": sample.target.canonical_label,
            "answer_text": sample.target.answer_text,
            "decision": sample.target.decision,
        }
    )


def audit_manifest(
    manifest_path: Path,
    *,
    target_format: str,
    short_threshold: int,
    max_examples_per_flag: int,
) -> dict[str, Any]:
    samples = read_manifest(manifest_path)
    by_task = Counter(sample.task_type for sample in samples)
    by_source = Counter(sample.source_dataset for sample in samples)
    by_task_source = Counter("%s::%s" % (sample.task_type, sample.source_dataset) for sample in samples)
    rendered_lengths: list[int] = []
    answer_lengths: list[int] = []
    label_counter: Counter[str] = Counter()
    answer_counter: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    flag_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    flag_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    examples_by_flag: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        rendered_target = target_to_text(sample, target_format=target_format)
        answer = answer_after_prefix(rendered_target)
        rendered_token_count = token_count(rendered_target)
        answer_token_count = token_count(answer)
        rendered_lengths.append(rendered_token_count)
        answer_lengths.append(answer_token_count)
        normalized_answer = normalize_text(answer).strip(".:")
        if normalized_answer:
            answer_counter[normalized_answer] += 1
        if sample.target.canonical_label:
            label_counter[normalize_text(sample.target.canonical_label)] += 1

        flags: list[str] = []
        if not rendered_target.strip():
            flags.append("empty_rendered_target")
        if rendered_target.strip().lower() == "answer:":
            flags.append("bare_answer_prefix")
        if sample.task_type in {"classification", "vqa"} and not rendered_target.startswith("Answer: "):
            flags.append("missing_answer_prefix")
        if sample.task_type in {"classification", "vqa"} and answer_token_count <= short_threshold:
            flags.append("short_answer")
        if normalized_answer in GENERIC_ANSWER_TERMS:
            flags.append("generic_answer")
        if sample.task_type == "classification" and not sample.target.canonical_label:
            flags.append("classification_missing_canonical_label")
        if sample.task_type == "vqa" and not sample.target.answer_text:
            flags.append("vqa_missing_answer_text")
        if sample.task_type == "clarify_or_respond" and not sample.target.decision:
            flags.append("clarify_missing_decision")
        if sample.task_type == "consultation":
            for section in ["Diagnosis:", "Evidence:", "Uncertainty:", "Management:", "Follow-up:"]:
                if section not in rendered_target:
                    flags.append("consultation_missing_%s" % section.rstrip(":").lower())
        if sample.task_type == "classification" and re.match(r"^\d+\s+", str(sample.target.canonical_label or "")):
            flags.append("classification_numeric_label_prefix")

        for flag in sorted(set(flags)):
            flag_counts[flag] += 1
            flag_by_task[flag][sample.task_type] += 1
            flag_by_source[flag][sample.source_dataset] += 1
            add_example(
                examples_by_flag,
                flag,
                sample,
                rendered_target,
                limit=max_examples_per_flag,
            )

    def length_summary(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "min": 0, "median": 0, "mean": 0.0, "max": 0}
        return {
            "count": len(values),
            "min": min(values),
            "median": median(values),
            "mean": mean(values),
            "max": max(values),
        }

    return {
        "manifest_path": str(manifest_path),
        "target_format": target_format,
        "num_rows": len(samples),
        "by_task": dict(sorted(by_task.items())),
        "by_source_dataset": dict(sorted(by_source.items())),
        "by_task_source_top30": dict(by_task_source.most_common(30)),
        "rendered_target_token_lengths": length_summary(rendered_lengths),
        "answer_token_lengths": length_summary(answer_lengths),
        "flag_counts": dict(sorted(flag_counts.items())),
        "flag_rates": {
            flag: count / float(len(samples)) if samples else 0.0 for flag, count in sorted(flag_counts.items())
        },
        "flag_by_task": {flag: dict(counter) for flag, counter in sorted(flag_by_task.items())},
        "flag_by_source_dataset": {flag: dict(counter.most_common(20)) for flag, counter in sorted(flag_by_source.items())},
        "top_rendered_answers": dict(answer_counter.most_common(50)),
        "top_classification_labels": dict(label_counter.most_common(50)),
        "classification_label_entropy": entropy(label_counter),
        "examples_by_flag": dict(examples_by_flag),
    }


def md_cell(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
    return text if text else "_empty_"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SFT Target Quality Audit",
        "",
        "- Manifest: `%s`" % payload["manifest_path"],
        "- Rows: `%s`" % payload["num_rows"],
        "- Target format: `%s`" % payload["target_format"],
        "",
        "## Task Mix",
        "",
        "| Task | Rows |",
        "| --- | ---: |",
    ]
    for task, count in payload["by_task"].items():
        lines.append("| %s | %s |" % (md_cell(task), count))

    lines.extend(
        [
            "",
            "## Target Lengths",
            "",
            "| Field | Count | Min | Median | Mean | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ["rendered_target_token_lengths", "answer_token_lengths"]:
        summary = payload[name]
        lines.append(
            "| %s | %s | %s | %s | %.3f | %s |"
            % (
                name,
                summary["count"],
                summary["min"],
                summary["median"],
                summary["mean"],
                summary["max"],
            )
        )

    lines.extend(["", "## Flags", "", "| Flag | Count | Rate |", "| --- | ---: | ---: |"])
    for flag, count in payload["flag_counts"].items():
        lines.append("| %s | %s | %.6f |" % (md_cell(flag), count, payload["flag_rates"][flag]))

    lines.extend(["", "## Top Rendered Answers", "", "| Answer | Count |", "| --- | ---: |"])
    for answer, count in list(payload["top_rendered_answers"].items())[:30]:
        lines.append("| %s | %s |" % (md_cell(answer), count))

    for flag, examples in payload["examples_by_flag"].items():
        lines.extend(["", "## Examples: `%s`" % flag, ""])
        lines.append("| # | Dataset | Task | Sample ID | Target |")
        lines.append("| ---: | --- | --- | --- | --- |")
        for index, example in enumerate(examples[:10], start=1):
            lines.append(
                "| %s | %s | %s | `%s` | %s |"
                % (
                    index,
                    md_cell(example["source_dataset"]),
                    md_cell(example["task_type"]),
                    md_cell(example["sample_id"]),
                    md_cell(example["target"]),
                )
            )

    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = audit_manifest(
        Path(args.manifest),
        target_format=args.target_format,
        short_threshold=args.short_answer_token_threshold,
        max_examples_per_flag=args.max_examples_per_flag,
    )
    write_json(Path(args.output_json), payload)
    write_markdown(Path(args.output_md), payload)
    print(json.dumps({key: payload[key] for key in ["manifest_path", "num_rows", "flag_counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
