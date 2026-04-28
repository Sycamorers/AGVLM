#!/usr/bin/env python3
"""Build non-overlapping SFT train and validation manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from agri_vlm.data.builders import build_sft_train_eval_manifests
from agri_vlm.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config for train/eval manifest construction.")
    return parser.parse_args()


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    payload = load_yaml(Path(args.config))
    summary = build_sft_train_eval_manifests(
        source_manifest_path=_resolve_path(repo_root, payload["source_manifest_path"]),
        holdout_manifest_path=_resolve_path(repo_root, payload["holdout_manifest_path"]),
        train_output_path=_resolve_path(repo_root, payload["train_output_path"]),
        eval_output_path=_resolve_path(repo_root, payload["eval_output_path"]),
        train_splits=payload["train_splits"],
        eval_splits=payload["eval_splits"],
        max_images_per_sample=payload["max_images_per_sample"],
        eval_sample_size=payload["eval_sample_size"],
        min_eval_samples_per_stratum=payload["min_eval_samples_per_stratum"],
        salt=payload["salt"],
        summary_output_path=_resolve_path(repo_root, payload["summary_output_path"]),
    )
    print(
        "Built SFT train/eval manifests: train_rows=%s eval_rows=%s eval_pool_rows=%s"
        % (summary["train_rows"], summary["eval_rows"], summary["eval_pool_rows"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
