#!/usr/bin/env python3
"""Attach closed-label metadata to an evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from agri_vlm.data.builders import build_closed_label_eval_manifest
from agri_vlm.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config for closed-label eval manifest construction.")
    return parser.parse_args()


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _str_list(payload: Dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ValueError("%s must be a list when set" % key)
    return [str(item) for item in value]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = load_yaml(Path(args.config))
    summary = build_closed_label_eval_manifest(
        input_manifest_path=_resolve_path(repo_root, config["input_manifest_path"]),
        label_space_manifest_path=_resolve_path(repo_root, config["label_space_manifest_path"]),
        output_manifest_path=_resolve_path(repo_root, config["output_manifest_path"]),
        summary_output_path=_resolve_path(repo_root, config["summary_output_path"]),
        strip_leading_numeric_prefix_sources=_str_list(config, "strip_leading_numeric_prefix_sources"),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
