#!/usr/bin/env python3
"""Normalize Agri-LLaVA-style records for the active subset tag."""

from __future__ import annotations

import argparse
from pathlib import Path

from agri_vlm.data.pipeline import normalize_dataset_spec, resolve_runtime_settings
from agri_vlm.data.registry import load_dataset_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/datasets.yaml")
    parser.add_argument("--download-mode", choices=["partial", "full"], default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--subset-tag", default=None)
    parser.add_argument("--data-root", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_dataset_registry(repo_root / args.config)
    runtime = resolve_runtime_settings(registry, repo_root, args.subset_tag, args.download_mode, args.fraction, args.data_root)
    rows = normalize_dataset_spec(
        spec=registry.specs["agrillava"],
        registry=registry,
        repo_root=repo_root,
        subset_tag=runtime["subset_tag"],
        data_root=str(runtime["data_root"]),
        download_mode=runtime["download_mode"],
        sample_fraction=runtime["sample_fraction"],
    )
    print("normalized agrillava rows=%s" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
