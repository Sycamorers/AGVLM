#!/bin/bash -l
#SBATCH --job-name=agri-vlm-baseline
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=benchmarks/vlm_baselines/logs/slurm/%x-%j.out
#SBATCH --error=benchmarks/vlm_baselines/logs/slurm/%x-%j.err

set -euo pipefail

cd /blue/hmedeiros/qinruoyao/agvlm

module load conda
module load cuda/12.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-agri-vlm-v1}"

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export AGRI_VLM_DATA_ROOT="$PWD/data"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TMPDIR="${TMPDIR:-$PWD/benchmarks/vlm_baselines/tmp/${SLURM_JOB_ID:-local}}"
export PYTHONPATH="$PWD/benchmarks/vlm_baselines${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

mkdir -p \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_XET_CACHE" \
  "$TMPDIR" \
  benchmarks/vlm_baselines/logs/slurm \
  benchmarks/vlm_baselines/results

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-bf16}"
QUANTIZATION="${QUANTIZATION:-none}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/vlm_baselines/results}"
RUN_ALL="${RUN_ALL:-0}"
MODEL_NAMES="${MODEL_NAMES:-}"
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"

"${PYTHON_BIN}" benchmarks/vlm_baselines/split_dataset.py

run_model() {
  local model_name="$1"
  local status_dir="$OUTPUT_DIR/status"
  mkdir -p "$status_dir"
  local args=(
    benchmarks/vlm_baselines/run_baselines.py
    --model-name "$model_name"
    --split "$SPLIT"
    --batch-size 1
    --max-new-tokens "$MAX_NEW_TOKENS"
    --dtype "$DTYPE"
    --quantization "$QUANTIZATION"
    --device "$DEVICE"
    --output-dir "$OUTPUT_DIR"
  )
  if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
    args+=(--attn-implementation "$ATTN_IMPLEMENTATION")
  fi
  if [[ "$MAX_SAMPLES" != "0" ]]; then
    args+=(--max-samples "$MAX_SAMPLES")
  fi
  echo "Running baseline model: $model_name"
  set +e
  srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "${PYTHON_BIN}" "${args[@]}"
  local exit_code=$?
  set -e
  printf '%s\t%s\t%s\n' "$(date -Iseconds)" "$exit_code" "$model_name" >> "$status_dir/model_status.tsv"
  if [[ "$exit_code" -ne 0 ]]; then
    echo "Baseline model failed with exit ${exit_code}: $model_name"
  fi
  return "$exit_code"
}

if [[ -n "$MODEL_NAMES" ]]; then
  overall_exit=0
  model_names_expanded="${MODEL_NAMES//,/ }"
  for model in $model_names_expanded; do
    if ! run_model "$model"; then
      overall_exit=1
    fi
  done
  exit "$overall_exit"
elif [[ "$RUN_ALL" == "1" ]]; then
  models=(
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    "google/paligemma2-3b-mix-448"
    "microsoft/Phi-4-multimodal-instruct"
    "allenai/Molmo2-4B"
    "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    "Qwen/Qwen2.5-VL-3B-Instruct"
  )
  overall_exit=0
  for model in "${models[@]}"; do
    if ! run_model "$model"; then
      overall_exit=1
    fi
  done
  exit "$overall_exit"
else
  run_model "$MODEL_NAME"
fi
