#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DTYPE="${DTYPE:-bf16}"
QUANTIZATION="${QUANTIZATION:-none}"
DEVICE="${DEVICE:-cuda:0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/vlm_baselines/results}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$PWD/benchmarks/vlm_baselines${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON_BIN}" benchmarks/vlm_baselines/split_dataset.py

args=(
  benchmarks/vlm_baselines/run_baselines.py
  --model-name "${MODEL_NAME}"
  --split "${SPLIT}"
  --batch-size 1
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --dtype "${DTYPE}"
  --quantization "${QUANTIZATION}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${MAX_SAMPLES}" != "0" ]]; then
  args+=(--max-samples "${MAX_SAMPLES}")
fi

"${PYTHON_BIN}" "${args[@]}"
