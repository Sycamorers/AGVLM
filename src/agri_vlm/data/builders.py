"""Manifest builders for SFT, RL, and evaluation."""

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from agri_vlm.data.manifest_io import (
    filter_rewardable_manifest,
    merge_manifests,
    read_manifest,
    write_manifest,
)
from agri_vlm.data.split_utils import assign_holdout
from agri_vlm.utils.io import read_jsonl, write_json


def build_sft_manifest(
    source_paths: Sequence[Path],
    output_path: Path,
    allowed_task_types: Sequence[str],
    exclude_splits: Sequence[str],
    max_samples_per_source: int = None,
) -> List[dict]:
    rows = merge_manifests(
        source_paths=source_paths,
        allowed_task_types=allowed_task_types,
        exclude_splits=exclude_splits,
        max_samples_per_source=max_samples_per_source,
    )
    return [sample.model_dump(mode="json") for sample in write_manifest(output_path, [row.model_dump(mode="json") for row in rows])]


def _manifest_group_key(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    images = row.get("images") or []
    source_image_id = metadata.get("source_image_id") or (images[0] if images else "")
    return "%s::%s" % (row.get("source_dataset"), source_image_id)


def _stable_hex(value: str, salt: str) -> str:
    return hashlib.sha256(("%s::%s" % (salt, value)).encode("utf-8")).hexdigest()


def _stratum_key(row: Dict[str, Any], fields: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def _counter_dict(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(Counter(str(row.get(field, "")) for row in rows))


def _sample_stratified(
    rows: Sequence[Dict[str, Any]],
    *,
    target_size: int,
    stratum_fields: Sequence[str],
    min_per_stratum: int,
    salt: str,
) -> List[Dict[str, Any]]:
    if target_size <= 0 or len(rows) <= target_size:
        return sorted(rows, key=lambda row: _stable_hex(str(row.get("sample_id")), salt))

    strata: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[_stratum_key(row, stratum_fields)].append(row)
    for key, stratum_rows in strata.items():
        strata[key] = sorted(stratum_rows, key=lambda row: _stable_hex(str(row.get("sample_id")), "%s::%s" % (salt, key)))

    allocations = {key: 0 for key in strata}
    remaining = target_size
    for key in sorted(strata, key=lambda item: len(strata[item]), reverse=True):
        if remaining <= 0:
            break
        take = min(min_per_stratum, len(strata[key]), remaining)
        allocations[key] = take
        remaining -= take

    if remaining > 0:
        capacities = {key: len(rows_for_key) - allocations[key] for key, rows_for_key in strata.items()}
        total_capacity = sum(max(value, 0) for value in capacities.values())
        fractional = []
        for key, capacity in capacities.items():
            if capacity <= 0 or total_capacity <= 0:
                continue
            raw = remaining * capacity / total_capacity
            take = min(capacity, int(raw))
            allocations[key] += take
            fractional.append((raw - take, capacity - take, key))
        remaining = target_size - sum(allocations.values())
        for _fraction, capacity, key in sorted(fractional, reverse=True):
            if remaining <= 0:
                break
            if capacity <= 0:
                continue
            allocations[key] += 1
            remaining -= 1

    sampled = []
    for key in sorted(strata):
        sampled.extend(strata[key][: allocations[key]])
    return sorted(sampled, key=lambda row: _stable_hex(str(row.get("sample_id")), salt))


def build_sft_train_eval_manifests(
    *,
    source_manifest_path: Path,
    holdout_manifest_path: Path,
    train_output_path: Path,
    eval_output_path: Path,
    train_splits: Sequence[str],
    eval_splits: Sequence[str],
    max_images_per_sample: int,
    eval_sample_size: int,
    min_eval_samples_per_stratum: int,
    salt: str,
    summary_output_path: Path = None,
) -> Dict[str, Any]:
    """Build non-overlapping SFT train and step-time validation manifests."""
    holdout_rows = [
        row
        for row in read_jsonl(holdout_manifest_path)
        if len(row.get("images") or []) <= max_images_per_sample
    ]
    eval_rows_by_id = {row["sample_id"]: row for row in holdout_rows}

    source_rows = list(read_jsonl(source_manifest_path))
    for row in source_rows:
        if len(row.get("images") or []) > max_images_per_sample:
            continue
        if row.get("split") in eval_splits:
            eval_rows_by_id.setdefault(row["sample_id"], row)

    eval_pool_rows = list(eval_rows_by_id.values())
    eval_ids = {row["sample_id"] for row in eval_pool_rows}
    eval_group_keys = {_manifest_group_key(row) for row in eval_pool_rows}

    train_rows = []
    excluded = Counter()
    for row in source_rows:
        if len(row.get("images") or []) > max_images_per_sample:
            excluded["max_images"] += 1
            continue
        if row.get("split") not in train_splits:
            excluded["split"] += 1
            continue
        if row["sample_id"] in eval_ids or _manifest_group_key(row) in eval_group_keys:
            excluded["eval_overlap"] += 1
            continue
        train_rows.append(row)

    if not train_rows:
        raise ValueError("No SFT training rows remain after train/eval split construction.")
    if not eval_pool_rows:
        raise ValueError("No SFT evaluation rows were selected for validation.")

    eval_rows = _sample_stratified(
        eval_pool_rows,
        target_size=eval_sample_size,
        stratum_fields=["source_dataset", "task_type", "split"],
        min_per_stratum=min_eval_samples_per_stratum,
        salt=salt,
    )

    train_ids = {row["sample_id"] for row in train_rows}
    train_group_keys = {_manifest_group_key(row) for row in train_rows}
    exact_overlap = sorted(train_ids.intersection(row["sample_id"] for row in eval_pool_rows))
    group_overlap = sorted(train_group_keys.intersection(_manifest_group_key(row) for row in eval_pool_rows))
    if exact_overlap or group_overlap:
        raise ValueError(
            "Train/eval overlap remains after split construction: exact=%s group=%s"
            % (len(exact_overlap), len(group_overlap))
        )

    write_manifest(train_output_path, train_rows)
    write_manifest(eval_output_path, eval_rows)
    summary = {
        "source_manifest_path": str(source_manifest_path),
        "holdout_manifest_path": str(holdout_manifest_path),
        "train_output_path": str(train_output_path),
        "eval_output_path": str(eval_output_path),
        "train_rows": len(train_rows),
        "eval_pool_rows": len(eval_pool_rows),
        "eval_rows": len(eval_rows),
        "train_splits": list(train_splits),
        "eval_splits": list(eval_splits),
        "max_images_per_sample": max_images_per_sample,
        "eval_sample_size": eval_sample_size,
        "min_eval_samples_per_stratum": min_eval_samples_per_stratum,
        "excluded": dict(excluded),
        "train_by_dataset": _counter_dict(train_rows, "source_dataset"),
        "train_by_task_type": _counter_dict(train_rows, "task_type"),
        "eval_by_dataset": _counter_dict(eval_rows, "source_dataset"),
        "eval_by_task_type": _counter_dict(eval_rows, "task_type"),
        "eval_by_split": _counter_dict(eval_rows, "split"),
        "overlap": {"exact_sample_id": 0, "group_key": 0},
    }
    if summary_output_path:
        write_json(summary_output_path, summary)
    return summary


def build_rl_manifest(
    source_paths: Sequence[Path],
    output_path: Path,
    allowed_task_types: Sequence[str],
    exclude_splits: Sequence[str],
    allowed_verifier_modes: Sequence[str],
    max_answer_words: int,
    max_images_per_sample: int = None,
) -> List[dict]:
    merged_rows = merge_manifests(
        source_paths=source_paths,
        allowed_task_types=allowed_task_types,
        exclude_splits=exclude_splits,
    )
    rewardable_rows = filter_rewardable_manifest(
        merged_rows,
        allowed_verifier_modes=allowed_verifier_modes,
        max_answer_words=max_answer_words,
    )
    if max_images_per_sample is not None:
        rewardable_rows = [row for row in rewardable_rows if len(row.images) <= max_images_per_sample]
    return [
        sample.model_dump(mode="json")
        for sample in write_manifest(output_path, [row.model_dump(mode="json") for row in rewardable_rows])
    ]


def build_eval_manifests(
    source_paths: Dict[str, Path],
    output_paths: Dict[str, Path],
    holdout_ratio: float,
    holdout_datasets: Sequence[str],
    salt: str,
) -> Dict[str, int]:
    summary = {}

    mirage_rows = read_manifest(source_paths["mirage"]) if source_paths.get("mirage") else []
    mmst_rows = []
    mmmt_rows = []
    for row in mirage_rows:
        track = str(row.metadata.get("benchmark_track") or "").lower()
        if "mmmt" in track or row.task_type == "clarify_or_respond":
            mmmt_rows.append(row)
        else:
            mmst_rows.append(row)
    write_manifest(output_paths["mirage_mmst"], [row.model_dump(mode="json") for row in mmst_rows])
    write_manifest(output_paths["mirage_mmmt"], [row.model_dump(mode="json") for row in mmmt_rows])
    summary["mirage_mmst"] = len(mmst_rows)
    summary["mirage_mmmt"] = len(mmmt_rows)

    holdout_rows = []
    fallback_rows = []
    for dataset_name in holdout_datasets:
        source_path = source_paths.get(dataset_name)
        if not source_path:
            continue
        for row in read_manifest(source_path):
            if row.split == "test":
                continue
            group_key = str(row.metadata.get("source_image_id") or row.images[0])
            payload = row.model_dump(mode="json")
            payload["split"] = "holdout"
            fallback_rows.append(payload)
            if not assign_holdout("%s::%s" % (dataset_name, group_key), salt=salt, holdout_ratio=holdout_ratio):
                continue
            holdout_rows.append(payload)
    if not holdout_rows and fallback_rows:
        holdout_rows.append(fallback_rows[0])
    write_manifest(output_paths["local_holdout"], holdout_rows)
    summary["local_holdout"] = len(holdout_rows)
    return summary
