"""Path helpers for dataset preparation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


DATA_ROOT_ENV = "AGRI_VLM_DATA_ROOT"


def get_data_root(repo_root: Path, explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get(DATA_ROOT_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (repo_root / "data").resolve()


def normalize_download_mode(value: str) -> str:
    normalized = (value or "partial").strip().lower()
    if normalized not in {"partial", "full"}:
        raise ValueError("Unsupported download_mode: %s" % value)
    return normalized


def normalize_sample_fraction(value: float) -> float:
    fraction = float(value)
    if fraction <= 0.0:
        raise ValueError("sample_fraction must be > 0, got %s" % value)
    if fraction > 1.0:
        raise ValueError("sample_fraction must be <= 1, got %s" % value)
    return fraction


def compute_subset_tag(download_mode: str, sample_fraction: float) -> str:
    normalized_mode = normalize_download_mode(download_mode)
    normalized_fraction = normalize_sample_fraction(sample_fraction)
    if normalized_mode == "full" or normalized_fraction >= 1.0:
        return "full"
    sample_percent = int(round(normalized_fraction * 100))
    return "partial_%spct" % sample_percent


def build_template_context(
    repo_root: Path,
    data_root: Path,
    subset_tag: str,
    dataset_name: str = "",
    download_mode: str = "partial",
    sample_fraction: float = 0.1,
) -> Dict[str, str]:
    fraction = normalize_sample_fraction(sample_fraction)
    sample_percent = int(round(fraction * 100))
    return {
        "repo_root": str(repo_root.resolve()),
        "data_root": str(data_root.resolve()),
        "subset_tag": subset_tag,
        "dataset_name": dataset_name,
        "download_mode": normalize_download_mode(download_mode),
        "sample_fraction": ("%s" % fraction),
        "sample_percent": str(sample_percent),
    }


def render_path_template(template: str, context: Dict[str, str]) -> Path:
    formatted = str(template).format(**context)
    return Path(formatted).expanduser().resolve()
