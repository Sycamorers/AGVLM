#!/usr/bin/env python3
"""Build the merged SFT manifest for a subset tag."""

from __future__ import annotations

import argparse
from pathlib import Path

from agri_vlm.data.builders import build_sft_manifest
from agri_vlm.data.pipeline import existing_interim_paths, resolve_manifest_path, resolve_runtime_settings
from agri_vlm.data.registry import load_dataset_registry
from agri_vlm.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/sft_build.yaml")
    parser.add_argument("--dataset-config", default="configs/data/datasets.yaml")
    parser.add_argument("--download-mode", choices=["partial", "full"], default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--subset-tag", default=None)
    parser.add_argument("--data-root", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_dataset_registry(repo_root / args.dataset_config)
    runtime = resolve_runtime_settings(
        registry=registry,
        repo_root=repo_root,
        subset_tag=args.subset_tag,
        download_mode=args.download_mode,
        sample_fraction=args.fraction,
        data_root=args.data_root,
    )
    payload = load_yaml(repo_root / args.config)
    source_paths = existing_interim_paths(
        registry=registry,
        dataset_names=payload["datasets"],
        repo_root=repo_root,
        subset_tag=runtime["subset_tag"],
        data_root=str(runtime["data_root"]),
        download_mode=runtime["download_mode"],
        sample_fraction=runtime["sample_fraction"],
    )
    output_path = resolve_manifest_path(
        template=payload["output_path"],
        repo_root=repo_root,
        subset_tag=runtime["subset_tag"],
        data_root=runtime["data_root"],
        download_mode=runtime["download_mode"],
        sample_fraction=runtime["sample_fraction"],
    )
    rows = build_sft_manifest(
        source_paths=list(source_paths.values()),
        output_path=output_path,
        allowed_task_types=payload["allowed_task_types"],
        exclude_splits=payload["exclude_splits"],
        max_samples_per_source=payload.get("max_samples_per_source"),
    )
    missing = [dataset_name for dataset_name in payload["datasets"] if dataset_name not in source_paths]
    print("built_sft_manifest=%s rows=%s" % (output_path, len(rows)))
    if missing:
        print("missing_sft_sources=%s" % ",".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
