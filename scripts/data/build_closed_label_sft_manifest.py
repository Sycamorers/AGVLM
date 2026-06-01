#!/usr/bin/env python3
"""Build a closed-label, per-class-balanced SFT manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from agri_vlm.data.builders import build_closed_label_sft_manifest
from agri_vlm.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config for closed-label SFT manifest construction.")
    return parser.parse_args()


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _positive_int_mapping(payload: Dict[str, Any], key: str) -> Dict[str, int]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError("%s must be a mapping when set" % key)
    parsed = {}
    for item_key, item_value in value.items():
        if not isinstance(item_value, int) or item_value < 1:
            raise ValueError("%s.%s must be a positive integer" % (key, item_key))
        parsed[str(item_key)] = item_value
    return parsed


def _required_positive_int_mapping(payload: Dict[str, Any], key: str) -> Dict[str, int]:
    parsed = _positive_int_mapping(payload, key)
    if not parsed:
        raise ValueError("%s must be a non-empty mapping" % key)
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
    config = load_yaml(Path(args.config))

    summary = build_closed_label_sft_manifest(
        input_manifest_path=_resolve_path(repo_root, config["input_manifest_path"]),
        output_manifest_path=_resolve_path(repo_root, config["output_manifest_path"]),
        summary_output_path=_resolve_path(repo_root, config["summary_output_path"]),
        classification_per_label_target=int(config["classification_per_label_target"]),
        classification_per_label_target_by_source=_positive_int_mapping(
            config,
            "classification_per_label_target_by_source",
        ),
        strip_leading_numeric_prefix_sources=[
            str(source) for source in config.get("strip_leading_numeric_prefix_sources", [])
        ],
        task_targets=_required_positive_int_mapping(config, "task_targets"),
        stratify_fields_by_task=_str_list_mapping(config, "stratify_fields_by_task"),
        min_per_stratum_by_task=_positive_int_mapping(config, "min_per_stratum_by_task"),
        seed=int(config.get("seed", 45)),
        shuffle=bool(config.get("shuffle", True)),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
