#!/usr/bin/env python3
"""Run GRPO post-training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agri_vlm.schemas.config_schema import ModelConfigSchema, RLTrainConfigSchema, load_config
from agri_vlm.training.rl_trainer import run_rl_grpo
from agri_vlm.utils.distributed import rank_zero_print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model/qwen_vlm_4b.yaml")
    parser.add_argument("--train-config", default="configs/train/rl_grpo_lora.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    model_config = load_config(repo_root / args.model_config, ModelConfigSchema)
    train_config = load_config(repo_root / args.train_config, RLTrainConfigSchema)
    if args.dry_run:
        train_config.dry_run = True
    if args.resume_from_checkpoint:
        train_config.resume_from_checkpoint = args.resume_from_checkpoint
    summary = run_rl_grpo(model_config=model_config, train_config=train_config)
    rank_zero_print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
