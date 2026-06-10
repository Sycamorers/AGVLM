#!/usr/bin/env python3
"""Build a filtered SFT manifest from an existing normalized manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Dict

from agri_vlm.utils.io import load_yaml, read_jsonl, write_json, write_jsonl


DEFAULT_DROP_FIELDS = {"benchmark_phase", "benchmark_split", "benchmark_split_policy", "phase"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config for filtered SFT manifest construction.")
    return parser.parse_args()


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _str_set(config: Dict[str, Any], key: str) -> set[str]:
    values = config.get(key) or []
    if not isinstance(values, list):
        raise ValueError("%s must be a list when set" % key)
    return {str(value) for value in values}


def _passes(row: dict[str, Any], *, allowed_task_types: set[str], sources: set[str], max_images_per_sample: int | None) -> bool:
    if allowed_task_types and str(row.get("task_type") or "") not in allowed_task_types:
        return False
    if sources and str(row.get("source_dataset") or "") not in sources:
        return False
    if max_images_per_sample is not None and len(row.get("images") or []) > max_images_per_sample:
        return False
    return True


def _clean_row(row: dict[str, Any], drop_fields: set[str]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in drop_fields}


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = load_yaml(Path(args.config))
    input_path = _resolve_path(repo_root, config["input_manifest_path"])
    output_path = _resolve_path(repo_root, config["output_manifest_path"])
    summary_path = _resolve_path(repo_root, config["summary_output_path"])

    allowed_task_types = _str_set(config, "allowed_task_types")
    required_task_types = _str_set(config, "required_task_types")
    sources = _str_set(config, "sources")
    drop_fields = DEFAULT_DROP_FIELDS | _str_set(config, "drop_fields")
    max_images = config.get("max_images_per_sample")
    max_images_per_sample = int(max_images) if max_images is not None else None
    if max_images_per_sample is not None and max_images_per_sample < 1:
        raise ValueError("max_images_per_sample must be positive when set.")

    rows = list(read_jsonl(input_path))
    selected = [
        _clean_row(row, drop_fields)
        for row in rows
        if _passes(
            row,
            allowed_task_types=allowed_task_types,
            sources=sources,
            max_images_per_sample=max_images_per_sample,
        )
    ]
    if not selected:
        raise ValueError("No rows selected from %s." % input_path)

    selected_task_types = {str(row.get("task_type") or "") for row in selected}
    missing_required = sorted(required_task_types - selected_task_types)
    if missing_required:
        raise ValueError("Filtered manifest is missing required task types: %s" % missing_required)

    if bool(config.get("shuffle", True)):
        rng = random.Random(int(config.get("seed", 57)))
        selected = list(selected)
        rng.shuffle(selected)

    write_jsonl(output_path, selected)
    summary = {
        "input_manifest_path": str(input_path),
        "output_manifest_path": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(selected),
        "allowed_task_types": sorted(allowed_task_types),
        "required_task_types": sorted(required_task_types),
        "sources": sorted(sources),
        "drop_fields": sorted(drop_fields),
        "max_images_per_sample": max_images_per_sample,
        "by_task_type": dict(Counter(str(row.get("task_type") or "") for row in selected)),
        "by_source_dataset": dict(Counter(str(row.get("source_dataset") or "") for row in selected)),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
