#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_KEY="${MODEL_KEY:-}"
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DTYPE="${DTYPE:-bf16}"
QUANTIZATION="${QUANTIZATION:-none}"
DEVICE="${DEVICE:-cuda:0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/vlm_baselines/results}"
DRY_RUN="${DRY_RUN:-0}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$PWD/benchmarks/vlm_baselines${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON_BIN}" benchmarks/vlm_baselines/build_phase_splits.py --phase rl --write-report

args=(
  benchmarks/vlm_baselines/run_baselines.py
  --phase rl
  --split "${SPLIT}"
  --batch-size 1
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --dtype "${DTYPE}"
  --quantization "${QUANTIZATION}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MODEL_KEY}" ]]; then
  args+=(--model-key "${MODEL_KEY}")
else
  args+=(--model-name "${MODEL_NAME}")
fi
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  args+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

"${PYTHON_BIN}" "${args[@]}"
