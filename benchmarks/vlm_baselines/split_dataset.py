#!/usr/bin/env python3
"""Create or refresh isolated benchmark split manifests.

The default path reuses the repository's active no-overlap SFT evaluation
manifest. If that manifest is absent, this script falls back to a deterministic
group-aware split of the available normalized SFT manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
from typing import Any

from dataset_adapter import (
    ACTIVE_EVAL_MANIFEST,
    ACTIVE_SPLIT_SUMMARY,
    ACTIVE_TRAIN_MANIFEST,
    FALLBACK_SOURCE_MANIFESTS,
    distribution_report,
    group_key,
    report_markdown,
    stratum_key,
)
from utils import BENCHMARK_ROOT, REPO_ROOT, ensure_dir, read_jsonl, stable_hash, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT / "splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _clone_for_benchmark(row: dict[str, Any], benchmark_split: str, policy: str) -> dict[str, Any]:
    cloned = copy.deepcopy(row)
    cloned["benchmark_split"] = benchmark_split
    cloned["benchmark_split_policy"] = policy
    return cloned


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(
    output_dir: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    policy: dict[str, Any],
    *,
    skipped: dict[str, int] | None = None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    for split, rows in rows_by_split.items():
        write_jsonl(output_dir / ("%s_manifest.jsonl" % split), rows)
    write_json(output_dir / "split_policy.json", policy)
    report = distribution_report(
        rows_by_split,
        skipped=skipped,
        extra={
            "policy": policy,
            "active_training_summary": _read_json_if_exists(REPO_ROOT / ACTIVE_SPLIT_SUMMARY),
            "active_train_manifest": str(ACTIVE_TRAIN_MANIFEST),
        },
    )
    write_json(output_dir / "distribution_report.json", report)
    (output_dir / "distribution_report.md").write_text(report_markdown(report), encoding="utf-8")
    return report


def _create_official_reuse_splits(output_dir: Path, seed: int) -> dict[str, Any]:
    source_path = REPO_ROOT / ACTIVE_EVAL_MANIFEST
    rows = read_jsonl(source_path)
    test_rows = [_clone_for_benchmark(row, "test", "official_active_eval_alias") for row in rows]
    val_source_rows = [row for row in rows if row.get("split") == "validation"]
    if not val_source_rows:
        val_source_rows = [row for row in rows if row.get("split") in {"validation", "dev"}]
    val_rows = [_clone_for_benchmark(row, "val", "official_validation_subset") for row in val_source_rows]
    policy = {
        "mode": "reuse_existing_held_out_eval",
        "seed": seed,
        "source_eval_manifest": str(ACTIVE_EVAL_MANIFEST),
        "test_manifest": "test_manifest.jsonl",
        "val_manifest": "val_manifest.jsonl",
        "notes": [
            "The repository has an active no-overlap held-out eval manifest, not a source split named test.",
            "Benchmark split 'test' is an alias for the full active 512-row held-out eval manifest.",
            "Benchmark split 'val' contains rows from the official source split tag 'validation'.",
            "No training manifest, dataset file, checkpoint, tokenizer, LoRA adapter, log, or output directory is modified.",
        ],
    }
    return _write_outputs(output_dir, {"test": test_rows, "val": val_rows}, policy)


def _source_manifest_for_fallback() -> Path:
    for relative in FALLBACK_SOURCE_MANIFESTS:
        candidate = REPO_ROOT / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No official active eval manifest and no fallback normalized SFT manifest found. Checked: %s"
        % ", ".join(str(path) for path in [ACTIVE_EVAL_MANIFEST, *FALLBACK_SOURCE_MANIFESTS])
    )


def _assign_grouped_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    group_strata: dict[str, str] = {}
    groups_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group, group_rows in groups.items():
        counter = Counter(stratum_key(row) for row in group_rows)
        stratum = counter.most_common(1)[0][0]
        group_strata[group] = stratum
        groups_by_stratum[stratum].append(group)

    assignments: dict[str, str] = {}
    for stratum, group_ids in sorted(groups_by_stratum.items()):
        ordered = sorted(group_ids, key=lambda value: stable_hash("%s|%s" % (stratum, value), seed=seed))
        total_rows = sum(len(groups[group]) for group in ordered)
        test_target = total_rows * test_ratio
        val_target = total_rows * val_ratio
        split_counts = {"test": 0, "val": 0, "train": 0}
        for group in ordered:
            size = len(groups[group])
            if split_counts["test"] < test_target:
                split = "test"
            elif split_counts["val"] < val_target:
                split = "val"
            else:
                split = "train"
            assignments[group] = split
            split_counts[split] += size
    return assignments


def _create_fallback_splits(
    output_dir: Path,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    source_path = _source_manifest_for_fallback()
    rows = read_jsonl(source_path)
    assignments = _assign_grouped_split(
        rows,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    rows_by_split = {"train": [], "val": [], "test": []}
    for row in rows:
        split = assignments[group_key(row)]
        rows_by_split[split].append(_clone_for_benchmark(row, split, "deterministic_group_stratified"))

    policy = {
        "mode": "created_deterministic_group_stratified_split",
        "seed": seed,
        "source_manifest": str(source_path.relative_to(REPO_ROOT)),
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "group_key_priority": [
            "scene_id",
            "dialogue_id/conversation_id",
            "image_id/video_id",
            "subject_id/participant_id",
            "source_image_id",
            "source file stem",
        ],
        "stratification": "source_dataset + task_type + verifier mode + label/answer",
        "notes": [
            "This fallback is used only when the repository active held-out eval manifest is absent.",
            "Baselines should be run only on val/test manifests.",
        ],
    }
    return _write_outputs(output_dir, rows_by_split, policy)


def ensure_split_manifests(
    output_dir: Path,
    *,
    seed: int = 42,
    force: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
) -> dict[str, Any]:
    if output_dir.exists() and not force and (output_dir / "distribution_report.json").exists():
        return json.loads((output_dir / "distribution_report.json").read_text(encoding="utf-8"))
    if (REPO_ROOT / ACTIVE_EVAL_MANIFEST).exists():
        return _create_official_reuse_splits(output_dir=output_dir, seed=seed)
    return _create_fallback_splits(
        output_dir=output_dir,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report = ensure_split_manifests(
        output_dir=output_dir,
        seed=args.seed,
        force=args.force,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
