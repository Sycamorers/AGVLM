#!/usr/bin/env python3
"""Build classification-only train/eval manifests for overfit probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from agri_vlm.data.builders import build_classification_probe_manifests
from agri_vlm.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config for classification probe construction.")
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
    summary = build_classification_probe_manifests(
        train_source_manifest_path=_resolve_path(repo_root, config["train_source_manifest_path"]),
        eval_source_manifest_path=_resolve_path(repo_root, config["eval_source_manifest_path"]),
        train_output_path=_resolve_path(repo_root, config["train_output_path"]),
        eval_output_path=_resolve_path(repo_root, config["eval_output_path"]),
        summary_output_path=_resolve_path(repo_root, config["summary_output_path"]),
        train_per_label=int(config["train_per_label"]),
        eval_per_label=int(config["eval_per_label"]),
        max_labels_per_source=int(config["max_labels_per_source"]),
        min_train_per_label=int(config.get("min_train_per_label", 1)),
        min_eval_per_label=int(config.get("min_eval_per_label", 1)),
        sources=_str_list(config, "sources"),
        strip_leading_numeric_prefix_sources=_str_list(config, "strip_leading_numeric_prefix_sources"),
        choice_format=str(config.get("choice_format", "label_list")),
        seed=int(config.get("seed", 53)),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
