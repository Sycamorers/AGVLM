#!/usr/bin/env python3
"""Run MIRAGE-MMST evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agri_vlm.evaluation.mirage_eval import run_mirage_eval_bundle
from agri_vlm.schemas.config_schema import EvalConfigSchema, ModelConfigSchema, load_config
from agri_vlm.utils.io import ensure_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model/qwen_vlm_4b.yaml")
    parser.add_argument("--eval-config", default="configs/eval/mirage_mmst.yaml")
    parser.add_argument("--prediction-mode", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--predictions-output", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    model_config = load_config(repo_root / args.model_config, ModelConfigSchema)
    eval_config = load_config(repo_root / args.eval_config, EvalConfigSchema)
    if args.prediction_mode:
        eval_config.prediction_mode = args.prediction_mode
    if args.checkpoint_path:
        eval_config.checkpoint_path = args.checkpoint_path
    if args.predictions_output:
        eval_config.predictions_path = args.predictions_output
    if args.max_examples is not None:
        eval_config.max_examples = args.max_examples
    result = run_mirage_eval_bundle(model_config=model_config, eval_config=eval_config)
    metrics = result["metrics"]
    output_path = Path(eval_config.output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if eval_config.predictions_path:
        write_jsonl(Path(eval_config.predictions_path), result["predictions"])
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
