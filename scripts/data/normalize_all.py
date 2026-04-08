#!/usr/bin/env python3
"""Normalize all materialized datasets for a subset tag."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--datasets", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_dataset_registry(repo_root / args.config)
    runtime = resolve_runtime_settings(
        registry=registry,
        repo_root=repo_root,
        subset_tag=args.subset_tag,
        download_mode=args.download_mode,
        sample_fraction=args.fraction,
        data_root=args.data_root,
    )
    selected_names = args.datasets or list(registry.specs.keys())

    summary = {}
    for dataset_name in selected_names:
        spec = registry.specs[dataset_name]
        try:
            rows = normalize_dataset_spec(
                spec=spec,
                registry=registry,
                repo_root=repo_root,
                subset_tag=runtime["subset_tag"],
                data_root=str(runtime["data_root"]),
                download_mode=runtime["download_mode"],
                sample_fraction=runtime["sample_fraction"],
            )
            summary[dataset_name] = {"status": "normalized", "rows": len(rows)}
        except FileNotFoundError as exc:
            summary[dataset_name] = {"status": "missing_raw", "reason": str(exc)}
        except Exception as exc:
            summary[dataset_name] = {"status": "error", "reason": str(exc)}
            raise

    print(
        json.dumps(
            {
                "subset_tag": runtime["subset_tag"],
                "download_mode": runtime["download_mode"],
                "sample_fraction": runtime["sample_fraction"],
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
