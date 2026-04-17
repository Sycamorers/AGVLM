#!/usr/bin/env python3
"""Run a reproducible evaluation suite for baseline or fine-tuned checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict

from agri_vlm.evaluation.local_eval import run_local_eval_bundle
from agri_vlm.evaluation.mirage_eval import run_mirage_eval_bundle
from agri_vlm.schemas.config_schema import EvalConfigSchema, ModelConfigSchema, load_config
from agri_vlm.utils.io import ensure_dir, write_json, write_jsonl


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "local_holdout": {
        "config": "configs/eval/local_holdout_full.yaml",
        "runner": run_local_eval_bundle,
    },
    "mirage_mmst": {
        "config": "configs/eval/mirage_mmst_full.yaml",
        "runner": run_mirage_eval_bundle,
    },
    "mirage_mmmt": {
        "config": "configs/eval/mirage_mmmt_full.yaml",
        "runner": run_mirage_eval_bundle,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model/qwen_vlm_4b.yaml")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_SPECS),
        default=["local_holdout"],
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--prediction-mode", default="model")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _run_task(
    *,
    repo_root: Path,
    model_config: Any,
    task_name: str,
    checkpoint_path: str | None,
    prediction_mode: str,
    batch_size: int | None,
    max_new_tokens: int | None,
    max_examples: int,
    output_dir: Path,
) -> dict:
    spec = TASK_SPECS[task_name]
    eval_config = load_config(repo_root / spec["config"], EvalConfigSchema)
    eval_config.prediction_mode = prediction_mode
    eval_config.checkpoint_path = checkpoint_path
    if batch_size is not None:
        eval_config.batch_size = batch_size
    if max_new_tokens is not None:
        eval_config.max_new_tokens = max_new_tokens
    if max_examples:
        eval_config.max_examples = max_examples

    task_output_dir = ensure_dir(output_dir / task_name)
    eval_config.output_path = str(task_output_dir / "metrics.json")
    eval_config.predictions_path = str(task_output_dir / "predictions.jsonl")

    runner: Callable[[Any, Any], dict] = spec["runner"]
    result = runner(model_config=model_config, eval_config=eval_config)
    write_json(Path(eval_config.output_path), result["metrics"])
    write_jsonl(Path(eval_config.predictions_path), result["predictions"])
    return {
        "task": task_name,
        "manifest_path": eval_config.manifest_path,
        "metrics_path": eval_config.output_path,
        "predictions_path": eval_config.predictions_path,
        "num_predictions": len(result["predictions"]),
        "metrics": result["metrics"],
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    model_config = load_config(repo_root / args.model_config, ModelConfigSchema)
    output_dir = ensure_dir(Path(args.output_dir))

    summary = {
        "model_config": args.model_config,
        "base_model_name_or_path": model_config.model_name_or_path,
        "checkpoint_path": args.checkpoint_path,
        "prediction_mode": args.prediction_mode,
        "tasks": [],
    }
    for task_name in args.tasks:
        summary["tasks"].append(
            _run_task(
                repo_root=repo_root,
                model_config=model_config,
                task_name=task_name,
                checkpoint_path=args.checkpoint_path,
                prediction_mode=args.prediction_mode,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                max_examples=args.max_examples,
                output_dir=output_dir,
            )
        )

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
