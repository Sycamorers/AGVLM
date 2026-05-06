#!/usr/bin/env python3
"""Build an SFT manifest that is disjoint from prior train/eval manifests."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON in %s line %s" % (path, line_number)) from exc


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sample_group_key(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    source_image_id = metadata.get("source_image_id")
    if not source_image_id:
        images = row.get("images") or []
        source_image_id = images[0] if images else ""
    return "%s::%s" % (row.get("source_dataset", ""), source_image_id)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config must be a mapping: %s" % path)
    return payload


def _as_path_list(config: Dict[str, Any], key: str) -> List[Path]:
    values = config.get(key, [])
    if not isinstance(values, list):
        raise ValueError("%s must be a list" % key)
    return [Path(str(value)) for value in values]


def _load_blocklists(paths: List[Path]) -> Tuple[set[str], set[str], Dict[str, Any]]:
    blocked_ids: set[str] = set()
    blocked_groups: set[str] = set()
    summaries = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError("Exclude manifest does not exist: %s" % path)
        ids_before = len(blocked_ids)
        groups_before = len(blocked_groups)
        rows = 0
        for row in _read_jsonl(path):
            rows += 1
            sample_id = row.get("sample_id")
            if sample_id:
                blocked_ids.add(str(sample_id))
            blocked_groups.add(_sample_group_key(row))
        summaries.append(
            {
                "path": str(path),
                "rows": rows,
                "new_sample_ids": len(blocked_ids) - ids_before,
                "new_group_keys": len(blocked_groups) - groups_before,
            }
        )
    return blocked_ids, blocked_groups, {"exclude_manifests": summaries}


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = _load_yaml(config_path)
    source_path = Path(str(config["source_manifest_path"]))
    output_path = Path(str(config["output_manifest_path"]))
    summary_path = Path(str(config["summary_output_path"]))
    exclude_paths = _as_path_list(config, "exclude_manifest_paths")
    max_images_per_sample = config.get("max_images_per_sample")
    allowed_task_types = config.get("allowed_task_types")
    allowed_source_datasets = config.get("allowed_source_datasets")
    allowed_splits = config.get("allowed_splits")
    seed = int(config.get("seed", 17))
    shuffle = bool(config.get("shuffle", True))

    if max_images_per_sample is not None:
        max_images_per_sample = int(max_images_per_sample)
        if max_images_per_sample < 1:
            raise ValueError("max_images_per_sample must be >= 1 when set")
    if allowed_task_types is not None:
        if not isinstance(allowed_task_types, list) or not allowed_task_types:
            raise ValueError("allowed_task_types must be a non-empty list when set")
        allowed_task_types = {str(item) for item in allowed_task_types}
    if allowed_source_datasets is not None:
        if not isinstance(allowed_source_datasets, list) or not allowed_source_datasets:
            raise ValueError("allowed_source_datasets must be a non-empty list when set")
        allowed_source_datasets = {str(item) for item in allowed_source_datasets}
    if allowed_splits is not None:
        if not isinstance(allowed_splits, list) or not allowed_splits:
            raise ValueError("allowed_splits must be a non-empty list when set")
        allowed_splits = {str(item) for item in allowed_splits}

    if not source_path.exists():
        raise FileNotFoundError("Source manifest does not exist: %s" % source_path)
    blocked_ids, blocked_groups, exclude_summary = _load_blocklists(exclude_paths)

    selected = []
    source_rows = 0
    excluded_by_reason: Counter[str] = Counter()
    output_tasks: Counter[str] = Counter()
    output_datasets: Counter[str] = Counter()
    output_splits: Counter[str] = Counter()
    output_image_counts: Counter[str] = Counter()
    for row in _read_jsonl(source_path):
        source_rows += 1
        sample_id = str(row.get("sample_id", ""))
        task_type = str(row.get("task_type", ""))
        image_count = len(row.get("images") or [])
        if sample_id in blocked_ids:
            excluded_by_reason["sample_id_overlap"] += 1
            continue
        if _sample_group_key(row) in blocked_groups:
            excluded_by_reason["group_key_overlap"] += 1
            continue
        if max_images_per_sample is not None and image_count > max_images_per_sample:
            excluded_by_reason["too_many_images"] += 1
            continue
        if allowed_task_types is not None and task_type not in allowed_task_types:
            excluded_by_reason["task_type_filtered"] += 1
            continue
        if allowed_source_datasets is not None and str(row.get("source_dataset", "")) not in allowed_source_datasets:
            excluded_by_reason["source_dataset_filtered"] += 1
            continue
        if allowed_splits is not None and str(row.get("split", "")) not in allowed_splits:
            excluded_by_reason["split_filtered"] += 1
            continue
        selected.append(row)
        output_tasks[task_type] += 1
        output_datasets[str(row.get("source_dataset", ""))] += 1
        output_splits[str(row.get("split", ""))] += 1
        output_image_counts[str(image_count)] += 1

    if not selected:
        raise ValueError("Disjoint manifest is empty after filtering: %s" % output_path)
    if shuffle:
        random.Random(seed).shuffle(selected)
    output_rows = _write_jsonl(output_path, selected)
    summary = {
        "config_path": str(config_path),
        "source_manifest_path": str(source_path),
        "output_manifest_path": str(output_path),
        "source_rows": source_rows,
        "output_rows": output_rows,
        "blocked_sample_ids": len(blocked_ids),
        "blocked_group_keys": len(blocked_groups),
        "max_images_per_sample": max_images_per_sample,
        "allowed_task_types": sorted(allowed_task_types) if allowed_task_types else None,
        "allowed_source_datasets": sorted(allowed_source_datasets) if allowed_source_datasets else None,
        "allowed_splits": sorted(allowed_splits) if allowed_splits else None,
        "seed": seed,
        "shuffle": shuffle,
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
        "output_by_task_type": dict(sorted(output_tasks.items())),
        "output_by_dataset": dict(sorted(output_datasets.items())),
        "output_by_split": dict(sorted(output_splits.items())),
        "output_image_count_histogram": dict(sorted(output_image_counts.items())),
        **exclude_summary,
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
