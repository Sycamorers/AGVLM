#!/usr/bin/env bash
set -euo pipefail

export MODEL_KEY="${MODEL_KEY:-agvlm_phi4_sft_completed}"
export MODEL_NAME=""
export SPLIT="${SPLIT:-test}"
export OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/vlm_baselines/results}"

bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_one_model.sh
