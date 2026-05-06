#!/usr/bin/env python3
"""Combine staged SFT run outputs into one lineage manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Stage in name=run_dir form. Provide in chronological order.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _parse_stage(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--stage must be in name=run_dir form: %s" % value)
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("Stage name is empty: %s" % value)
    return name, Path(path)


def _checkpoint_number(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _find_checkpoints(run_dir: Path, metadata: Optional[Dict[str, Any]]) -> List[str]:
    candidates = [run_dir]
    if metadata and metadata.get("checkpoint_output_dir"):
        candidates.append(Path(str(metadata["checkpoint_output_dir"])))
    checkpoints = []
    for candidate in candidates:
        if candidate.exists():
            checkpoints.extend(path for path in candidate.glob("checkpoint-*") if path.is_dir())
    unique = {str(path): path for path in checkpoints}
    return [str(path) for path in sorted(unique.values(), key=_checkpoint_number)]


def _metrics_path(run_dir: Path) -> Optional[Path]:
    candidates = [run_dir / "metrics" / "train_metrics.jsonl", run_dir / "metrics.jsonl"]
    for path in candidates:
        if path.exists():
            return path
    return None


def _best_metric(rows: Iterable[Dict[str, Any]], key: str, *, higher_is_better: bool) -> Optional[Dict[str, Any]]:
    best = None
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if best is None:
            best = row
            continue
        previous = best[key]
        if higher_is_better and value > previous:
            best = row
        if not higher_is_better and value < previous:
            best = row
    if best is None:
        return None
    return {"global_step": best.get("global_step"), key: best[key]}


def _stage_summary(name: str, run_dir: Path) -> Dict[str, Any]:
    metadata = _read_json(run_dir / "run_metadata.json")
    dry_run_summary = _read_json(run_dir / "dry_run_summary.json")
    artifact_manifest = _read_json(run_dir / "artifact_manifest.json")
    metrics = []
    metrics_jsonl = _metrics_path(run_dir)
    if metrics_jsonl:
        metrics = _read_jsonl(metrics_jsonl)
    checkpoints = _find_checkpoints(run_dir, metadata)
    final_adapter_path = None
    checkpoint_output_dir = Path(str(metadata["checkpoint_output_dir"])) if metadata and metadata.get("checkpoint_output_dir") else run_dir
    if (checkpoint_output_dir / "adapter_model.safetensors").exists():
        final_adapter_path = str(checkpoint_output_dir)
    elif checkpoints:
        final_adapter_path = checkpoints[-1]

    return {
        "name": name,
        "run_dir": str(run_dir),
        "checkpoint_output_dir": str(checkpoint_output_dir),
        "metrics_jsonl": str(metrics_jsonl) if metrics_jsonl else None,
        "resolved_config": str(run_dir / "resolved_config.yaml") if (run_dir / "resolved_config.yaml").exists() else None,
        "run_metadata": str(run_dir / "run_metadata.json") if (run_dir / "run_metadata.json").exists() else None,
        "artifact_manifest": str(run_dir / "artifact_manifest.json") if (run_dir / "artifact_manifest.json").exists() else None,
        "exported_artifact_manifest": artifact_manifest,
        "dry_run_summary": dry_run_summary,
        "checkpoints": checkpoints,
        "final_adapter_path": final_adapter_path,
        "final_metric_row": metrics[-1] if metrics else None,
        "best_eval_loss": _best_metric(metrics, "eval_loss", higher_is_better=False),
        "best_answer_exact_match": _best_metric(
            metrics,
            "eval_performance_answer_exact_match",
            higher_is_better=True,
        ),
        "best_average_reward": _best_metric(
            metrics,
            "eval_performance_average_reward",
            higher_is_better=True,
        ),
    }


def main() -> int:
    args = parse_args()
    if not args.stage:
        raise ValueError("At least one --stage is required")
    stages = [_stage_summary(name, path) for name, path in map(_parse_stage, args.stage)]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lineage_order": [stage["name"] for stage in stages],
        "stages": stages,
        "recommended_final_adapter": stages[-1].get("final_adapter_path") if stages else None,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
