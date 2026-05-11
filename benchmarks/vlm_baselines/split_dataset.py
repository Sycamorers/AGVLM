#!/usr/bin/env python3
"""Backward-compatible SFT split wrapper.

New benchmark code should use ``build_phase_splits.py`` so SFT and RL benchmark
splits remain explicit. This wrapper preserves the original ``val_manifest`` /
``test_manifest`` outputs as aliases for the SFT benchmark phase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_phase_splits import build_phase_splits
from utils import BENCHMARK_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT / "splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-fallback-split", action="store_true")
    return parser.parse_args()


def ensure_split_manifests(
    output_dir: Path,
    *,
    seed: int = 42,
    force: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    allow_fallback_split: bool = False,
) -> dict[str, Any]:
    del train_ratio, val_ratio, test_ratio
    return build_phase_splits(
        phase="sft",
        output_dir=output_dir,
        seed=seed,
        force=force,
        allow_fallback_split=allow_fallback_split,
        write_report=True,
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report = ensure_split_manifests(
        output_dir=output_dir,
        seed=args.seed,
        force=args.force,
        allow_fallback_split=args.allow_fallback_split,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
