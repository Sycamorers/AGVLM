#!/usr/bin/env python3
"""Download supported dataset subsets into subset-tagged raw directories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from agri_vlm.data.hf_download import download_supported_datasets
from agri_vlm.data.pipeline import resolve_runtime_settings
from agri_vlm.data.registry import load_dataset_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/datasets.yaml")
    parser.add_argument("--download-mode", choices=["partial", "full"], default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--subset-tag", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
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
    summary = download_supported_datasets(
        registry=registry,
        repo_root=repo_root,
        subset_tag=runtime["subset_tag"],
        download_mode=runtime["download_mode"],
        sample_fraction=runtime["sample_fraction"],
        data_root=str(runtime["data_root"]),
        dataset_names=args.datasets,
        token=os.environ.get("HF_TOKEN"),
        dry_run=args.dry_run,
    )
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
    # The HF datasets/pyarrow stack can leave background state that aborts
    # during interpreter finalization on the HPC image after successful work.
    # Flush explicitly and exit without running global teardown hooks.
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
