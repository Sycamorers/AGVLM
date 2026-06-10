#!/usr/bin/env python3
"""Render prompt/target examples for an SFT manifest formatting audit."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from agri_vlm.data.conversation_format import (
    CLASSIFICATION_LABEL_ONLY_FORMAT,
    INSTRUCTIONAL_FORMAT,
    MANIFEST_PROMPT_FORMAT,
    PLAIN_FORMAT,
    sample_to_prompt_messages,
    target_to_text,
)
from agri_vlm.data.manifest_io import read_manifest
from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.io import ensure_dir, load_yaml, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config for the rendered format audit.")
    return parser.parse_args()


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _stable_hex(value: str, salt: str) -> str:
    return hashlib.sha256(("%s::%s" % (salt, value)).encode("utf-8")).hexdigest()


def _target_identity(sample: UnifiedSample) -> str:
    target = sample.target
    return "::".join(
        [
            target.answer_text or "",
            target.canonical_label or "",
            "|".join(target.canonical_labels),
            target.decision or "",
        ]
    )


def _sample_key(sample: UnifiedSample) -> str:
    return "%s::%s::%s" % (sample.sample_id, ",".join(sample.images), _target_identity(sample))


def _dedupe_samples(samples: Iterable[UnifiedSample]) -> List[UnifiedSample]:
    seen = set()
    output = []
    for sample in samples:
        key = _sample_key(sample)
        if key in seen:
            continue
        seen.add(key)
        output.append(sample)
    return output


def _first_user_text(messages: List[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        for content in message.get("content") or []:
            if content.get("type") == "text":
                return str(content.get("text") or "").strip()
    return ""


def _validation_failures(
    sample: UnifiedSample,
    prompt: str,
    target: str,
    *,
    prompt_format: str,
    target_format: str,
) -> List[str]:
    failures = []
    prompt_lower = prompt.lower()
    is_label_only_classification = (
        sample.task_type == "classification"
        and prompt_format == CLASSIFICATION_LABEL_ONLY_FORMAT
        and target_format == CLASSIFICATION_LABEL_ONLY_FORMAT
    )
    if (
        not is_label_only_classification
        and "answer:" not in prompt_lower
        and "decision:" not in prompt_lower
        and "diagnosis:" not in prompt_lower
    ):
        failures.append("prompt_missing_output_contract")

    if (
        sample.task_type in {"classification", "vqa"}
        and not is_label_only_classification
        and not target.startswith("Answer: ")
    ):
        failures.append("target_missing_answer_prefix")
    if is_label_only_classification:
        if re.search(r"(?im)^\s*(answer|evidence|choice)\s*:", target):
            failures.append("label_only_target_contains_structured_prefix")
        if "\n" in target:
            failures.append("label_only_target_contains_newline")
    if sample.task_type == "clarify_or_respond":
        if target.startswith("Decision: clarify"):
            if "\nClarifying question: " not in target:
                failures.append("clarify_target_missing_question")
        elif target.startswith("Decision: respond"):
            if "\nAnswer: " not in target:
                failures.append("respond_target_missing_answer")
        else:
            failures.append("clarify_target_missing_decision")
    if sample.task_type == "consultation":
        for section in ["Diagnosis:", "Evidence:", "Uncertainty:", "Management:", "Follow-up:"]:
            if section not in target:
                failures.append("consultation_target_missing_%s" % section.rstrip(":").lower())
    return failures


def _markdown_block(text: str) -> str:
    return "```text\n%s\n```" % text.replace("```", "'''").strip()


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# SFT Format Audit",
        "",
        "- Manifest: `%s`" % payload["manifest_path"],
        "- Prompt format: `%s`" % payload["prompt_format"],
        "- Target format: `%s`" % payload["target_format"],
        "- Unique examples rendered: `%s`" % payload["rendered_examples"],
        "- Validation failures: `%s`" % payload["validation_failure_count"],
        "",
        "## Counts",
        "",
        "| Task | Manifest rows | Rendered unique examples |",
        "| --- | ---: | ---: |",
    ]
    for task_type in sorted(payload["manifest_by_task"]):
        lines.append(
            "| %s | %s | %s |"
            % (
                task_type,
                payload["manifest_by_task"][task_type],
                payload["rendered_by_task"].get(task_type, 0),
            )
        )

    if payload["validation_failures"]:
        lines.extend(["", "## Validation Failures", ""])
        for name, count in payload["validation_failures"].items():
            lines.append("- `%s`: %s" % (name, count))

    grouped = defaultdict(list)
    for example in payload["examples"]:
        grouped[example["task_type"]].append(example)

    for task_type in sorted(grouped):
        lines.extend(["", "## %s" % task_type, ""])
        for index, example in enumerate(grouped[task_type], start=1):
            lines.extend(
                [
                    "### %s.%s `%s`" % (task_type, index, example["sample_id"]),
                    "",
                    "- Source: `%s`" % example["source_dataset"],
                    "- Verifier: `%s`" % example["verifier_mode"],
                    "- Images: `%s`" % "`, `".join(example["images"]),
                    "",
                    "**Prompt**",
                    "",
                    _markdown_block(example["prompt"]),
                    "",
                    "**Target**",
                    "",
                    _markdown_block(example["target"]),
                    "",
                ]
            )

    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def render_audit(config_path: Path) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    manifest_path = _resolve_path(repo_root, config["manifest_path"])
    markdown_output_path = _resolve_path(repo_root, config["markdown_output_path"])
    json_output_path = _resolve_path(repo_root, config["json_output_path"])
    samples_per_task = int(config.get("samples_per_task", 20))
    seed = int(config.get("seed", 41))
    prompt_format = str(config.get("prompt_format", INSTRUCTIONAL_FORMAT))
    target_format = str(config.get("target_format", INSTRUCTIONAL_FORMAT))

    if prompt_format not in {MANIFEST_PROMPT_FORMAT, INSTRUCTIONAL_FORMAT, CLASSIFICATION_LABEL_ONLY_FORMAT}:
        raise ValueError("Unsupported prompt_format: %s" % prompt_format)
    if target_format not in {PLAIN_FORMAT, INSTRUCTIONAL_FORMAT, CLASSIFICATION_LABEL_ONLY_FORMAT}:
        raise ValueError("Unsupported target_format: %s" % target_format)
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be >= 1")

    rows = read_manifest(manifest_path)
    manifest_by_task = Counter(sample.task_type for sample in rows)
    grouped = defaultdict(list)
    for sample in _dedupe_samples(rows):
        grouped[sample.task_type].append(sample)

    examples = []
    validation_failures = Counter()
    rendered_by_task = Counter()
    for task_type in sorted(grouped):
        ordered = sorted(grouped[task_type], key=lambda sample: _stable_hex(_sample_key(sample), "%s::%s" % (seed, task_type)))
        for sample in ordered[:samples_per_task]:
            prompt_messages = sample_to_prompt_messages(sample, prompt_format=prompt_format)
            prompt = _first_user_text(prompt_messages)
            target = target_to_text(sample, target_format=target_format)
            failures = _validation_failures(
                sample,
                prompt,
                target,
                prompt_format=prompt_format,
                target_format=target_format,
            )
            validation_failures.update(failures)
            rendered_by_task[task_type] += 1
            examples.append(
                {
                    "sample_id": sample.sample_id,
                    "source_dataset": sample.source_dataset,
                    "task_type": sample.task_type,
                    "split": sample.split,
                    "verifier_mode": sample.verifier.mode,
                    "images": sample.images,
                    "prompt": prompt,
                    "target": target,
                    "validation_failures": failures,
                }
            )

    payload = {
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "markdown_output_path": str(markdown_output_path),
        "json_output_path": str(json_output_path),
        "prompt_format": prompt_format,
        "target_format": target_format,
        "samples_per_task": samples_per_task,
        "seed": seed,
        "manifest_rows": len(rows),
        "manifest_by_task": dict(sorted(manifest_by_task.items())),
        "rendered_examples": len(examples),
        "rendered_by_task": dict(sorted(rendered_by_task.items())),
        "validation_failure_count": sum(validation_failures.values()),
        "validation_failures": dict(sorted(validation_failures.items())),
        "examples": examples,
    }
    write_json(json_output_path, payload)
    _write_markdown(markdown_output_path, payload)
    return payload


def main() -> int:
    args = parse_args()
    payload = render_audit(Path(args.config))
    print(
        "Rendered %s audit examples with %s validation failures: %s"
        % (
            payload["rendered_examples"],
            payload["validation_failure_count"],
            payload["markdown_output_path"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
