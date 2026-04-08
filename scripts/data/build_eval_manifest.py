#!/usr/bin/env python3
"""Build evaluation manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from agri_vlm.data.builders import build_eval_manifests
from agri_vlm.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/eval_build.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    payload = load_yaml(repo_root / args.config)
    summary = build_eval_manifests(
        source_paths={key: repo_root / value for key, value in payload["sources"].items()},
        output_paths={key: repo_root / value for key, value in payload["output_paths"].items()},
        holdout_ratio=payload["holdout_ratio"],
        holdout_datasets=payload["holdout_datasets"],
        salt=payload["salt"],
    )
    for key, value in sorted(summary.items()):
        print("built_%s=%s" % (key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
