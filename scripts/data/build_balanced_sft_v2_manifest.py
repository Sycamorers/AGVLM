#!/usr/bin/env python3
"""Build a capped and oversampled SFT manifest for the next SFT pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from agri_vlm.data.builders import build_balanced_sft_v2_manifest
from agri_vlm.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config for balanced SFT manifest construction.")
    return parser.parse_args()


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _positive_int_mapping(payload: Dict[str, Any], key: str) -> Dict[str, int]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError("%s must be a non-empty mapping" % key)
    parsed = {}
    for item_key, item_value in value.items():
        if not isinstance(item_value, int) or item_value < 1:
            raise ValueError("%s.%s must be a positive integer" % (key, item_key))
        parsed[str(item_key)] = item_value
    return parsed


def _str_list_mapping(payload: Dict[str, Any], key: str) -> Dict[str, list[str]]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError("%s must be a mapping when set" % key)
    parsed = {}
    for item_key, item_value in value.items():
        if not isinstance(item_value, list) or not item_value:
            raise ValueError("%s.%s must be a non-empty list" % (key, item_key))
        parsed[str(item_key)] = [str(item) for item in item_value]
    return parsed


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    config = load_yaml(config_path)

    summary = build_balanced_sft_v2_manifest(
        input_manifest_path=_resolve_path(repo_root, config["input_manifest_path"]),
        output_manifest_path=_resolve_path(repo_root, config["output_manifest_path"]),
        summary_output_path=_resolve_path(repo_root, config["summary_output_path"]),
        task_targets=_positive_int_mapping(config, "task_targets"),
        stratify_fields_by_task=_str_list_mapping(config, "stratify_fields_by_task"),
        min_per_stratum_by_task=_positive_int_mapping(config, "min_per_stratum_by_task")
        if config.get("min_per_stratum_by_task")
        else None,
        seed=int(config.get("seed", 41)),
        shuffle=bool(config.get("shuffle", True)),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
