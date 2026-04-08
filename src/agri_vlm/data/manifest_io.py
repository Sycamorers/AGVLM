"""Manifest read, write, merge, and filter helpers."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.io import ensure_dir, read_jsonl, write_jsonl
from agri_vlm.utils.text import word_count


def validate_rows(rows: Iterable[dict]) -> List[UnifiedSample]:
    return [UnifiedSample.model_validate(row) for row in rows]


def write_manifest(path: Path, rows: Iterable[dict]) -> List[UnifiedSample]:
    path = Path(path)
    validated = validate_rows(rows)
    ensure_dir(path.parent)
    write_jsonl(path, [sample.model_dump(mode="json") for sample in validated])
    return validated


def read_manifest(path: Path) -> List[UnifiedSample]:
    path = Path(path)
    return [UnifiedSample.model_validate(row) for row in read_jsonl(path)]


def merge_manifests(
    source_paths: Sequence[Path],
    allowed_task_types: Optional[Sequence[str]] = None,
    exclude_splits: Optional[Sequence[str]] = None,
    max_samples_per_source: Optional[int] = None,
) -> List[UnifiedSample]:
    merged = []
    for source_path in source_paths:
        count = 0
        for row in read_jsonl(source_path):
            sample = UnifiedSample.model_validate(row)
            if allowed_task_types and sample.task_type not in allowed_task_types:
                continue
            if exclude_splits and sample.split in exclude_splits:
                continue
            merged.append(sample)
            count += 1
            if max_samples_per_source and count >= max_samples_per_source:
                break
    return merged


def filter_rewardable_manifest(
    rows: Sequence[UnifiedSample], allowed_verifier_modes: Sequence[str], max_answer_words: int
) -> List[UnifiedSample]:
    filtered = []
    for sample in rows:
        if sample.verifier.mode not in allowed_verifier_modes:
            continue
        answer_text = sample.target.answer_text or sample.target.canonical_label or ""
        if answer_text and word_count(answer_text) > max_answer_words:
            continue
        filtered.append(sample)
    return filtered


def summarize_manifest(rows: Sequence[UnifiedSample]) -> Dict[str, Dict[str, int]]:
    summary = {"by_dataset": {}, "by_task_type": {}, "by_split": {}}
    for sample in rows:
        summary["by_dataset"][sample.source_dataset] = (
            summary["by_dataset"].get(sample.source_dataset, 0) + 1
        )
        summary["by_task_type"][sample.task_type] = (
            summary["by_task_type"].get(sample.task_type, 0) + 1
        )
        summary["by_split"][sample.split] = summary["by_split"].get(sample.split, 0) + 1
    return summary


def dump_summary(path: Path, rows: Sequence[UnifiedSample]) -> None:
    summary = summarize_manifest(rows)
    ensure_dir(path.parent)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
