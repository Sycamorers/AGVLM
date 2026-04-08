#!/usr/bin/env python3
"""Generate a dataset report for a subset tag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agri_vlm.data.pipeline import resolve_manifest_path, resolve_runtime_settings
from agri_vlm.data.registry import load_dataset_registry
from agri_vlm.data.reporting import build_dataset_report, write_dataset_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/data/datasets.yaml")
    parser.add_argument("--download-mode", choices=["partial", "full"], default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--subset-tag", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-json", default="{data_root}/manifests/{subset_tag}/dataset_report.json")
    parser.add_argument("--output-markdown", default="{data_root}/manifests/{subset_tag}/dataset_report.md")
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
    report = build_dataset_report(
        registry=registry,
        repo_root=repo_root,
        subset_tag=runtime["subset_tag"],
        data_root=str(runtime["data_root"]),
        download_mode=runtime["download_mode"],
        sample_fraction=runtime["sample_fraction"],
    )
    output_json = resolve_manifest_path(
        template=args.output_json,
        repo_root=repo_root,
        subset_tag=runtime["subset_tag"],
        data_root=runtime["data_root"],
        download_mode=runtime["download_mode"],
        sample_fraction=runtime["sample_fraction"],
    )
    output_markdown = resolve_manifest_path(
        template=args.output_markdown,
        repo_root=repo_root,
        subset_tag=runtime["subset_tag"],
        data_root=runtime["data_root"],
        download_mode=runtime["download_mode"],
        sample_fraction=runtime["sample_fraction"],
    )
    write_dataset_report(report, output_json, output_markdown)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
