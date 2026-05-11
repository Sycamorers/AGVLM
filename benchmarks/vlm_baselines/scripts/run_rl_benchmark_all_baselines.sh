#!/usr/bin/env bash
set -euo pipefail

models=(
  "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
  "google/paligemma2-3b-mix-448"
  "microsoft/Phi-4-multimodal-instruct"
  "allenai/Molmo2-4B"
  "llava-hf/llava-onevision-qwen2-7b-ov-hf"
  "Qwen/Qwen2.5-VL-3B-Instruct"
)

for model in "${models[@]}"; do
  MODEL_NAME="${model}" MODEL_KEY="" bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_one_model.sh
done
