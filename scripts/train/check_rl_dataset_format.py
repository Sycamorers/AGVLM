#!/usr/bin/env python3
"""Validate RL manifest rows before handing them to TRL GRPOTrainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.conversation_format import sample_to_prompt_messages
from agri_vlm.data.manifest_io import read_manifest
from agri_vlm.modeling.processor_factory import load_processor
from agri_vlm.rewards.composite import make_trl_reward_function
from agri_vlm.schemas.config_schema import ModelConfigSchema, load_config
from agri_vlm.utils.image import open_image
from agri_vlm.utils.io import ensure_dir, write_json


DATASET_COLUMNS = [
    "prompt",
    "image_paths",
    "task_type",
    "sample_id",
    "target_json",
    "verifier_json",
    "reward_meta_json",
    "metadata_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--check-processor", action="store_true")
    return parser.parse_args()


def _drop_none_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none_fields(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none_fields(item) for item in value]
    return value


def _prompt_content_stats(prompt: List[Dict[str, Any]]) -> Dict[str, int]:
    stats = {"messages": len(prompt), "text_blocks": 0, "image_blocks": 0}
    for message in prompt:
        for content in message.get("content") or []:
            if content.get("type") == "text" and str(content.get("text") or "").strip():
                stats["text_blocks"] += 1
            if content.get("type") == "image" and str(content.get("image") or "").strip():
                stats["image_blocks"] += 1
    return stats


def _record_from_sample(sample: Any) -> Dict[str, Any]:
    return {
        "prompt": _drop_none_fields(sample_to_prompt_messages(sample)),
        "image_paths": sample.images,
        "task_type": sample.task_type,
        "sample_id": sample.sample_id,
        "target_json": json.dumps(sample.target.model_dump(mode="json"), ensure_ascii=False),
        "verifier_json": json.dumps(sample.verifier.model_dump(mode="json"), ensure_ascii=False),
        "reward_meta_json": json.dumps(sample.reward_meta.model_dump(mode="json"), ensure_ascii=False),
        "metadata_json": json.dumps(sample.metadata, ensure_ascii=False),
    }


def run_format_check(
    *,
    manifest_path: Path,
    model_config_path: Path,
    max_samples: int,
    output_json: Path,
    output_md: Path,
    check_processor: bool = False,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    model_config = load_config(model_config_path, ModelConfigSchema)
    rows = read_manifest(manifest_path)
    if max_samples > 0:
        rows = rows[:max_samples]

    issue_examples: Dict[str, List[Dict[str, Any]]] = {}

    def add_issue(name: str, sample_id: str, reason: str) -> None:
        issue_examples.setdefault(name, [])
        if len(issue_examples[name]) < 20:
            issue_examples[name].append({"sample_id": sample_id, "reason": reason})

    records: List[Dict[str, Any]] = []
    prompt_stats: List[Dict[str, int]] = []
    for sample in rows:
        record = _record_from_sample(sample)
        records.append(record)
        if sorted(record.keys()) != sorted(DATASET_COLUMNS):
            add_issue("dataset_column_mismatch", sample.sample_id, "columns=%s" % sorted(record.keys()))
        stats = _prompt_content_stats(record["prompt"])
        prompt_stats.append(stats)
        if stats["messages"] == 0 or stats["text_blocks"] == 0:
            add_issue("invalid_prompt_messages", sample.sample_id, "missing text prompt block")
        if stats["image_blocks"] == 0:
            add_issue("missing_prompt_image_block", sample.sample_id, "prompt has no image content block")
        for image_path in record["image_paths"]:
            if not (repo_root / image_path).exists():
                add_issue("image_path_missing", sample.sample_id, image_path)
                break
        for json_key in ("target_json", "verifier_json", "reward_meta_json", "metadata_json"):
            try:
                json.loads(record[json_key])
            except json.JSONDecodeError:
                add_issue("json_serialization_error", sample.sample_id, json_key)

    reward_function_ok = False
    transformed_sample_check: Dict[str, Any] = {"ok": False, "images_key": "images", "image_count": 0}
    if records:
        reward_fn = make_trl_reward_function(
            reward_modules=["exact_match", "normalized_label", "clarify_vs_respond"],
            reward_weights={},
        )
        try:
            reward_fn(
                prompts=[records[0]["prompt"]],
                completions=[""],
                task_type=[records[0]["task_type"]],
                target_json=[records[0]["target_json"]],
                verifier_json=[records[0]["verifier_json"]],
                reward_meta_json=[records[0]["reward_meta_json"]],
            )
            reward_function_ok = True
        except Exception as exc:
            add_issue("reward_function_column_error", records[0]["sample_id"], str(exc))
        try:
            loaded_images = [open_image(repo_root / image_path) for image_path in records[0]["image_paths"]]
            transformed_sample_check = {
                "ok": bool(loaded_images),
                "images_key": "images",
                "image_count": len(loaded_images),
                "modes": [image.mode for image in loaded_images],
            }
        except Exception as exc:
            transformed_sample_check = {"ok": False, "images_key": "images", "image_count": 0, "error": str(exc)}
            add_issue("transformed_sample_image_error", records[0]["sample_id"], str(exc))

    processor_check: Dict[str, Any] = {"requested": bool(check_processor), "ok": None}
    if check_processor and records:
        try:
            processor = load_processor(model_config)
            rendered = processor.apply_chat_template(
                records[0]["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
            processor_check = {"requested": True, "ok": True, "rendered_type": type(rendered).__name__}
        except Exception as exc:
            processor_check = {"requested": True, "ok": False, "error": str(exc)}
            add_issue("processor_check_error", records[0]["sample_id"], str(exc))

    report = {
        "manifest_path": str(manifest_path),
        "model_config_path": str(model_config_path),
        "checked_rows": len(rows),
        "dataset_columns": DATASET_COLUMNS,
        "reward_function_columns_ok": reward_function_ok,
        "transformed_sample_check": transformed_sample_check,
        "prompt_content_summary": {
            "min_messages": min((item["messages"] for item in prompt_stats), default=0),
            "min_text_blocks": min((item["text_blocks"] for item in prompt_stats), default=0),
            "min_image_blocks": min((item["image_blocks"] for item in prompt_stats), default=0),
        },
        "processor_check": processor_check,
        "issues": {name: {"count": len(examples), "examples": examples} for name, examples in sorted(issue_examples.items())},
    }
    write_json(output_json, report)
    write_markdown_report(report, output_md)
    return report


def write_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# RL Dataset Format Check",
        "",
        "- Manifest: `%s`" % report["manifest_path"],
        "- Model config: `%s`" % report["model_config_path"],
        "- Checked rows: `%s`" % report["checked_rows"],
        "- Reward function columns OK: `%s`" % report["reward_function_columns_ok"],
        "",
        "## Dataset Columns",
        "",
    ]
    for column in report["dataset_columns"]:
        lines.append("- `%s`" % column)
    lines.extend(["", "## Prompt Content Summary", ""])
    for key, value in report["prompt_content_summary"].items():
        lines.append("- `%s`: `%s`" % (key, value))
    lines.extend(["", "## Processor Check", ""])
    for key, value in report["processor_check"].items():
        lines.append("- `%s`: `%s`" % (key, value))
    lines.extend(["", "## Transformed Sample Check", ""])
    for key, value in report["transformed_sample_check"].items():
        lines.append("- `%s`: `%s`" % (key, value))
    lines.extend(["", "## Issues", ""])
    if not report["issues"]:
        lines.append("No issues found.")
    for name, payload in report["issues"].items():
        lines.append("### %s" % name)
        lines.append("")
        lines.append("- Count shown: `%s`" % payload["count"])
        for example in payload["examples"]:
            lines.append("- `%s`: %s" % (example["sample_id"], example["reason"]))
        lines.append("")
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = run_format_check(
        manifest_path=Path(args.manifest_path),
        model_config_path=Path(args.model_config),
        max_samples=args.max_samples,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        check_processor=args.check_processor,
    )
    issue_count = sum(payload["count"] for payload in report["issues"].values())
    print("rl_dataset_format_check=%s checked_rows=%s issues=%s" % (
        args.output_json,
        report["checked_rows"],
        issue_count,
    ))
    return 0 if report["checked_rows"] and issue_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
