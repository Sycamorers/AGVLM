#!/usr/bin/env python3
"""Build an upsampled SFT manifest from explicit repeat factors."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from agri_vlm.data.manifest_io import write_manifest
from agri_vlm.utils.io import load_yaml, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _positive_int_mapping(payload: Dict[str, Any], key: str) -> Dict[str, int]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError("%s must be a non-empty mapping" % key)
    factors = {}
    for factor_key, factor_value in value.items():
        if not isinstance(factor_value, int) or factor_value < 1:
            raise ValueError("%s.%s must be a positive integer" % (key, factor_key))
        factors[str(factor_key)] = factor_value
    return factors


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    input_path = Path(config["input_manifest_path"])
    output_path = Path(config["output_manifest_path"])
    summary_path = Path(config["summary_output_path"])
    task_repeat_factors = _positive_int_mapping(config, "task_repeat_factors")
    seed = int(config.get("seed", 17))
    shuffle = bool(config.get("shuffle", True))

    rows = list(read_jsonl(input_path))
    if not rows:
        raise ValueError("Input manifest is empty: %s" % input_path)

    input_tasks = Counter(str(row.get("task_type")) for row in rows)
    missing_factors = sorted(set(input_tasks) - set(task_repeat_factors))
    unused_factors = sorted(set(task_repeat_factors) - set(input_tasks))
    if missing_factors or unused_factors:
        raise ValueError(
            "Task repeat factors must exactly cover input task types. missing=%s unused=%s"
            % (missing_factors, unused_factors)
        )

    expanded = []
    repeat_histogram = Counter()
    for row in rows:
        task_type = str(row["task_type"])
        repeat_factor = task_repeat_factors[task_type]
        repeat_histogram[task_type] += repeat_factor
        expanded.extend(row for _ in range(repeat_factor))

    if shuffle:
        random.Random(seed).shuffle(expanded)

    validated = write_manifest(output_path, expanded)
    output_tasks = Counter(sample.task_type for sample in validated)
    output_datasets = Counter(sample.source_dataset for sample in validated)
    summary = {
        "input_manifest_path": str(input_path),
        "output_manifest_path": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(validated),
        "seed": seed,
        "shuffle": shuffle,
        "task_repeat_factors": task_repeat_factors,
        "input_by_task_type": dict(sorted(input_tasks.items())),
        "output_by_task_type": dict(sorted(output_tasks.items())),
        "output_by_dataset": dict(sorted(output_datasets.items())),
    }
    write_json(summary_path, summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
