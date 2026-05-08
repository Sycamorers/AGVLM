#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
export SPLIT="${SPLIT:-val}"
export MAX_SAMPLES="${MAX_SAMPLES:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/vlm_baselines/results_smoke}"

bash benchmarks/vlm_baselines/scripts/run_one_model.sh
