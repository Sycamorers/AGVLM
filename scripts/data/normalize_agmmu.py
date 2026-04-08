#!/usr/bin/env python3
"""Normalize AgMMU exports into the unified JSONL schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from agri_vlm.data.manifest_io import write_manifest
from agri_vlm.data.normalizers import normalize_vqa_like_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/agmmu")
    parser.add_argument("--output", default="data/interim/agmmu.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    rows = normalize_vqa_like_dataset(
        raw_dir=repo_root / args.raw_dir,
        repo_root=repo_root,
        dataset_name="agmmu",
        default_task_type="vqa",
    )
    write_manifest(repo_root / args.output, rows)
    print("normalized agmmu rows=%s" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
