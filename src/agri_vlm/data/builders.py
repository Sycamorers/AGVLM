"""Manifest builders for SFT, RL, and evaluation."""

from pathlib import Path
from typing import Dict, List, Sequence

from agri_vlm.data.manifest_io import (
    filter_rewardable_manifest,
    merge_manifests,
    read_manifest,
    write_manifest,
)
from agri_vlm.data.split_utils import assign_holdout


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


def build_rl_manifest(
    source_paths: Sequence[Path],
    output_path: Path,
    allowed_task_types: Sequence[str],
    exclude_splits: Sequence[str],
    allowed_verifier_modes: Sequence[str],
    max_answer_words: int,
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

    agmmu_rows = read_manifest(source_paths["agmmu"]) if source_paths.get("agmmu") else []
    write_manifest(output_paths["agmmu"], [row.model_dump(mode="json") for row in agmmu_rows])
    summary["agmmu"] = len(agmmu_rows)

    agrobench_rows = read_manifest(source_paths["agrobench"]) if source_paths.get("agrobench") else []
    if output_paths.get("agrobench"):
        write_manifest(output_paths["agrobench"], [row.model_dump(mode="json") for row in agrobench_rows])
    summary["agrobench"] = len(agrobench_rows)

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
