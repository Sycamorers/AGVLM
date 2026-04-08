#!/usr/bin/env python3
"""Build evaluation manifests for a subset tag."""

from __future__ import annotations

import argparse
from pathlib import Path

from agri_vlm.data.builders import build_eval_manifests
from agri_vlm.data.pipeline import existing_interim_paths, resolve_manifest_path, resolve_runtime_settings
from agri_vlm.data.registry import load_dataset_registry
from agri_vlm.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/eval_build.yaml")
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
    dataset_names = list(payload["sources"].values())
    source_paths = existing_interim_paths(
        registry=registry,
        dataset_names=dataset_names,
        repo_root=repo_root,
        subset_tag=runtime["subset_tag"],
        data_root=str(runtime["data_root"]),
        download_mode=runtime["download_mode"],
        sample_fraction=runtime["sample_fraction"],
    )
    keyed_sources = {
        key: source_paths[dataset_name]
        for key, dataset_name in payload["sources"].items()
        if dataset_name in source_paths
    }
    output_paths = {
        key: resolve_manifest_path(
            template=value,
            repo_root=repo_root,
            subset_tag=runtime["subset_tag"],
            data_root=runtime["data_root"],
            download_mode=runtime["download_mode"],
            sample_fraction=runtime["sample_fraction"],
        )
        for key, value in payload["output_paths"].items()
    }
    summary = build_eval_manifests(
        source_paths=keyed_sources,
        output_paths=output_paths,
        holdout_ratio=payload["holdout_ratio"],
        holdout_datasets=payload["holdout_datasets"],
        salt=payload["salt"],
    )
    missing = [
        "%s:%s" % (key, dataset_name)
        for key, dataset_name in payload["sources"].items()
        if dataset_name not in source_paths
    ]
    for key, value in sorted(summary.items()):
        print("built_%s=%s" % (key, value))
    if missing:
        print("missing_eval_sources=%s" % ",".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
