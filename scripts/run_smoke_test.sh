#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PYTHONPATH:-src}"

"${PYTHON_BIN}" scripts/data/prepare_manual_dataset_slots.py --with-smoke-data --download-mode partial --fraction 0.1
"${PYTHON_BIN}" scripts/data/normalize_all.py --download-mode partial --fraction 0.1
"${PYTHON_BIN}" scripts/data/build_sft_manifest.py --config configs/data/sft_build.yaml --download-mode partial --fraction 0.1
"${PYTHON_BIN}" scripts/data/build_rl_manifest.py --config configs/data/rl_build.yaml --download-mode partial --fraction 0.1
"${PYTHON_BIN}" scripts/data/build_eval_manifest.py --config configs/data/eval_build.yaml --download-mode partial --fraction 0.1
"${PYTHON_BIN}" scripts/data/dataset_report.py --download-mode partial --fraction 0.1 >/dev/null
"${PYTHON_BIN}" scripts/train/train_sft.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_smoke_1gpu.yaml
"${PYTHON_BIN}" scripts/train/train_rl_grpo.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/rl_grpo_smoke_1gpu.yaml
"${PYTHON_BIN}" scripts/eval/eval_local_holdout.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --eval-config configs/eval/local_holdout.yaml \
  --prediction-mode oracle
"${PYTHON_BIN}" scripts/launch_torchrun.py --dry-run --nproc-per-node 1 \
  scripts/train/train_sft.py -- \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_smoke_1gpu.yaml >/dev/null

echo "Smoke pipeline completed."
