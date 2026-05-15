#!/usr/bin/env python3
"""Repair source-specific classification labels in an existing manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from agri_vlm.data.manifest_io import write_manifest
from agri_vlm.utils.io import load_yaml, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config for label repair.")
    return parser.parse_args()


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _clean_label(value: Any, *, strip_leading_numeric_prefix: bool) -> str:
    label = re.sub(r"\s+", " ", str(value or "").replace("_", " ").replace("-", " ")).strip()
    if strip_leading_numeric_prefix:
        label = re.sub(r"^\d+\s+", "", label).strip()
    return label


def _dedupe(values: Iterable[str]) -> List[str]:
    output = []
    seen = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(str(value).strip())
    return output


def _repair_row(
    row: Dict[str, Any],
    *,
    source_datasets: set[str],
    strip_leading_numeric_prefix: bool,
) -> tuple[Dict[str, Any], bool]:
    if row.get("task_type") != "classification" or row.get("source_dataset") not in source_datasets:
        return row, False

    target = dict(row.get("target") or {})
    verifier = dict(row.get("verifier") or {})
    old_label = target.get("canonical_label") or target.get("answer_text")
    new_label = _clean_label(old_label, strip_leading_numeric_prefix=strip_leading_numeric_prefix)
    if not old_label or not new_label or new_label == old_label:
        return row, False

    repaired = dict(row)
    repaired["target"] = target
    repaired["verifier"] = verifier
    repaired["metadata"] = dict(row.get("metadata") or {})
    repaired["metadata"].setdefault("original_canonical_label", old_label)
    repaired["metadata"]["label_repair"] = "strip_leading_numeric_prefix"

    target["canonical_label"] = new_label
    if target.get("answer_text") == old_label:
        target["answer_text"] = new_label
    if target.get("canonical_labels"):
        target["canonical_labels"] = _dedupe(
            [_clean_label(value, strip_leading_numeric_prefix=strip_leading_numeric_prefix) for value in target["canonical_labels"]]
        )

    verifier["accepted_labels"] = _dedupe(
        [new_label]
        + [
            _clean_label(value, strip_leading_numeric_prefix=strip_leading_numeric_prefix)
            for value in verifier.get("accepted_labels") or []
        ]
        + [str(value) for value in verifier.get("accepted_labels") or []]
        + [str(old_label)]
    )
    return repaired, True


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = load_yaml(Path(args.config))
    input_path = _resolve_path(repo_root, config["input_manifest_path"])
    output_path = _resolve_path(repo_root, config["output_manifest_path"])
    summary_path = _resolve_path(repo_root, config["summary_output_path"])
    source_datasets = {str(value) for value in config.get("source_datasets", [])}
    if not source_datasets:
        raise ValueError("source_datasets must be a non-empty list")
    strip_leading_numeric_prefix = bool(config.get("strip_leading_numeric_prefix", True))

    rows = []
    repaired_by_source: Counter[str] = Counter()
    numeric_prefix_after: Counter[str] = Counter()
    total_by_source: Counter[str] = Counter()
    for row in read_jsonl(input_path):
        repaired, changed = _repair_row(
            row,
            source_datasets=source_datasets,
            strip_leading_numeric_prefix=strip_leading_numeric_prefix,
        )
        rows.append(repaired)
        if repaired.get("task_type") == "classification":
            source = str(repaired.get("source_dataset") or "")
            total_by_source[source] += 1
            label = str((repaired.get("target") or {}).get("canonical_label") or "")
            if re.match(r"^\s*\d+\s+", label):
                numeric_prefix_after[source] += 1
        if changed:
            repaired_by_source[str(row.get("source_dataset") or "")] += 1

    validated = write_manifest(output_path, rows)
    summary = {
        "input_manifest_path": str(input_path),
        "output_manifest_path": str(output_path),
        "output_rows": len(validated),
        "source_datasets": sorted(source_datasets),
        "strip_leading_numeric_prefix": strip_leading_numeric_prefix,
        "repaired_by_source": dict(sorted(repaired_by_source.items())),
        "classification_rows_by_source": dict(sorted(total_by_source.items())),
        "numeric_prefix_after_by_source": dict(sorted(numeric_prefix_after.items())),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
