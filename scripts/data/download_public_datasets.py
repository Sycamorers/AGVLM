#!/usr/bin/env python3
"""Create manual dataset slots and optionally report mirror locations."""

from __future__ import annotations

import argparse
from pathlib import Path

from agri_vlm.data.registry import create_manual_slot, load_dataset_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data/datasets.yaml",
        help="Dataset registry configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_dataset_registry(repo_root / args.config)
    for spec in registry.values():
        slot_dir = create_manual_slot(spec, repo_root=repo_root)
        print("prepared_slot=%s" % slot_dir)
    print("Manual slots prepared. This script does not auto-download licensed or ambiguous sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
