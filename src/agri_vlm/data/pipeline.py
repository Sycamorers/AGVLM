"""High-level dataset prep helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from agri_vlm.data.loaders import find_first_records_file
from agri_vlm.data.manifest_io import read_manifest, write_manifest
from agri_vlm.data.normalizers import (
    load_provenance_metadata,
    normalize_classification_directory_dataset,
    normalize_classification_records_dataset,
    normalize_consultation_dataset,
    normalize_ip102_dataset,
    normalize_vqa_like_dataset,
)
from agri_vlm.data.paths import compute_subset_tag, get_data_root, render_path_template, build_template_context
from agri_vlm.data.registry import DatasetRegistry, DatasetSpec
from agri_vlm.utils.image import collect_image_paths


def resolve_runtime_settings(
    registry: DatasetRegistry,
    repo_root: Path,
    subset_tag: Optional[str] = None,
    download_mode: Optional[str] = None,
    sample_fraction: Optional[float] = None,
    data_root: Optional[str] = None,
) -> Dict[str, object]:
    actual_download_mode = download_mode or registry.defaults.download_mode
    actual_sample_fraction = sample_fraction or registry.defaults.sample_fraction
    actual_subset_tag = subset_tag or compute_subset_tag(actual_download_mode, actual_sample_fraction)
    actual_data_root = get_data_root(repo_root, data_root)
    return {
        "download_mode": actual_download_mode,
        "sample_fraction": actual_sample_fraction,
        "subset_tag": actual_subset_tag,
        "data_root": actual_data_root,
    }


def has_materialized_raw_data(raw_dir: Path) -> bool:
    if not raw_dir.exists():
        return False
    if find_first_records_file(raw_dir) is not None:
        return True
    return bool(collect_image_paths(raw_dir))


def normalize_dataset_spec(
    spec: DatasetSpec,
    registry: DatasetRegistry,
    repo_root: Path,
    subset_tag: str,
    data_root: Optional[str] = None,
    download_mode: Optional[str] = None,
    sample_fraction: Optional[float] = None,
) -> List[dict]:
    raw_dir = spec.raw_dir(
        repo_root=repo_root,
        defaults=registry.defaults,
        subset_tag=subset_tag,
        data_root=data_root,
        download_mode=download_mode,
        sample_fraction=sample_fraction,
    )
    if not has_materialized_raw_data(raw_dir):
        raise FileNotFoundError("No materialized raw dataset found for %s at %s" % (spec.name, raw_dir))

    provenance = load_provenance_metadata(raw_dir)
    common_kwargs = {
        "raw_dir": raw_dir,
        "repo_root": repo_root,
        "license_name": spec.license_name or None,
        "provenance": provenance,
    }
    if spec.normalizer == "classification_directory":
        rows = normalize_classification_directory_dataset(
            dataset_name=spec.name,
            salt="%s-v1" % spec.name,
            pest_mode=spec.name == "ip102",
            **common_kwargs,
        )
    elif spec.normalizer == "classification_records":
        rows = normalize_classification_records_dataset(
            dataset_name=spec.name,
            pest_mode=spec.name == "ip102",
            **common_kwargs,
        )
    elif spec.normalizer == "ip102":
        rows = normalize_ip102_dataset(**common_kwargs)
    elif spec.normalizer == "vqa_records":
        rows = normalize_vqa_like_dataset(
            dataset_name=spec.name,
            default_task_type=spec.default_task_type or "vqa",
            **common_kwargs,
        )
    elif spec.normalizer == "consultation_records":
        rows = normalize_consultation_dataset(dataset_name=spec.name, **common_kwargs)
    else:
        raise ValueError("Unsupported normalizer for %s: %s" % (spec.name, spec.normalizer))

    output_path = spec.interim_path(
        repo_root=repo_root,
        defaults=registry.defaults,
        subset_tag=subset_tag,
        data_root=data_root,
        download_mode=download_mode,
        sample_fraction=sample_fraction,
    )
    write_manifest(output_path, rows)
    return rows


def existing_interim_paths(
    registry: DatasetRegistry,
    dataset_names: Iterable[str],
    repo_root: Path,
    subset_tag: str,
    data_root: Optional[str] = None,
    download_mode: Optional[str] = None,
    sample_fraction: Optional[float] = None,
) -> Dict[str, Path]:
    paths = {}
    for dataset_name in dataset_names:
        spec = registry.specs[dataset_name]
        path = spec.interim_path(
            repo_root=repo_root,
            defaults=registry.defaults,
            subset_tag=subset_tag,
            data_root=data_root,
            download_mode=download_mode,
            sample_fraction=sample_fraction,
        )
        if path.exists():
            paths[dataset_name] = path
    return paths


def resolve_manifest_path(
    template: str,
    repo_root: Path,
    subset_tag: str,
    data_root: Path,
    download_mode: str,
    sample_fraction: float,
) -> Path:
    context = build_template_context(
        repo_root=repo_root,
        data_root=data_root,
        subset_tag=subset_tag,
        download_mode=download_mode,
        sample_fraction=sample_fraction,
    )
    return render_path_template(template, context)


def manifest_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_manifest(path))
