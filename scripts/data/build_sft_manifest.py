#!/usr/bin/env python3
"""Build the merged SFT manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from agri_vlm.data.builders import build_sft_manifest
from agri_vlm.schemas.config_schema import load_config, ManifestBuildConfigSchema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/sft_build.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    payload = load_config(repo_root / args.config, ManifestBuildConfigSchema)
    rows = build_sft_manifest(
        source_paths=[repo_root / path for path in payload.sources],
        output_path=repo_root / payload.output_path,
        allowed_task_types=payload.model_extra["allowed_task_types"],
        exclude_splits=payload.model_extra["exclude_splits"],
        max_samples_per_source=payload.model_extra.get("max_samples_per_source"),
    )
    print("built_sft_manifest=%s rows=%s" % (payload.output_path, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
