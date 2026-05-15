#!/usr/bin/env python3
"""Convert user-provided public agricultural QA rows into the RL manifest format.

This is a scaffold for additional licensed data. It does not download data or
hard-code dataset URLs; provide a local JSONL file plus source/license metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.manifest_io import write_manifest
from agri_vlm.utils.io import read_jsonl, write_json
from agri_vlm.utils.text import normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--default-split", default="train")
    parser.add_argument("--default-task-type", choices=["classification", "vqa", "consultation"], default="vqa")
    return parser.parse_args()


def _required_text(row: Dict[str, Any], keys: List[str], row_index: int) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise ValueError("Input row %s is missing one of %s." % (row_index, keys))


def _image_path(row: Dict[str, Any], image_root: Path, row_index: int) -> str:
    value = _required_text(row, ["image", "image_path", "file_name"], row_index)
    image_path = Path(value)
    resolved = image_path if image_path.is_absolute() else image_root / image_path
    if not resolved.exists():
        raise FileNotFoundError("Input row %s image does not exist: %s" % (row_index, resolved))
    return str(resolved)


def _manifest_row(
    row: Dict[str, Any],
    *,
    row_index: int,
    image_root: Path,
    source_name: str,
    source_license: str,
    source_url: str,
    default_split: str,
    default_task_type: str,
) -> Dict[str, Any]:
    task_type = str(row.get("task_type") or default_task_type)
    image = _image_path(row, image_root, row_index)
    question = _required_text(row, ["question", "prompt", "instruction"], row_index)
    answer = _required_text(row, ["answer", "answer_text", "response"], row_index)
    label = str(row.get("label") or row.get("canonical_label") or "").strip()
    normalized = normalize_label(label or answer)
    target: Dict[str, Any] = {"answer_text": answer, "acceptable_answers": [answer]}
    verifier: Dict[str, Any] = {"mode": "exact_match", "accepted_answers": [answer]}
    reward_weights = {"exact_match": 1.0, "hallucination_penalty": 1.0}
    if task_type == "classification":
        target["canonical_label"] = normalized
        verifier = {"mode": "label", "accepted_labels": [normalized]}
        reward_weights = {"normalized_label": 1.0, "hallucination_penalty": 1.0}
    return {
        "sample_id": str(row.get("sample_id") or "%s-%06d" % (source_name, row_index)),
        "source_dataset": source_name,
        "task_type": task_type,
        "split": str(row.get("split") or default_split),
        "images": [image],
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an agricultural vision-language assistant focused on ground-level RGB crop consultation.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ],
        "target": target,
        "metadata": {
            "source_type": "user_provided_public_scaffold",
            "source_url": source_url,
            "license": source_license,
            "crop": row.get("crop"),
            "disease": row.get("disease"),
            "normalized_label": normalized if task_type == "classification" else None,
        },
        "verifier": verifier,
        "reward_meta": {"weights": reward_weights},
    }


def main() -> int:
    args = parse_args()
    if not args.source_license.strip():
        raise ValueError("--source-license is required so gated/licensed data remains explicit.")
    rows = [
        _manifest_row(
            row,
            row_index=index,
            image_root=Path(args.image_root),
            source_name=args.source_name,
            source_license=args.source_license,
            source_url=args.source_url,
            default_split=args.default_split,
            default_task_type=args.default_task_type,
        )
        for index, row in enumerate(read_jsonl(Path(args.input_jsonl)), start=1)
    ]
    validated = write_manifest(Path(args.output), rows)
    summary_path = Path(args.summary_output) if args.summary_output else Path(args.output).with_suffix(".summary.json")
    write_json(
        summary_path,
        {
            "input_jsonl": args.input_jsonl,
            "output": args.output,
            "rows": len(validated),
            "source_name": args.source_name,
            "source_url": args.source_url,
            "source_license": args.source_license,
            "downloads_data": False,
            "validation": "UnifiedSample schema validation and image existence checks completed.",
        },
    )
    print("ingested_public_agri_manifest=%s rows=%s summary=%s" % (args.output, len(validated), summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
