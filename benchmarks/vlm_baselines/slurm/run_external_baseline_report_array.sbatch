#!/bin/bash -l
#SBATCH --job-name=agri-vlm-extbench-array
#SBATCH --partition=hpg-turin
#SBATCH --array=0-11%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=benchmarks/vlm_baselines/logs/slurm/%x-%A_%a.out
#SBATCH --error=benchmarks/vlm_baselines/logs/slurm/%x-%A_%a.err

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
export TMPDIR="$PWD/benchmarks/vlm_baselines/tmp/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}"
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
  benchmarks/vlm_baselines/logs/slurm

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/vlm_baselines/results/baseline_report_20260516}"
SPLIT="${SPLIT:-test}"
DTYPE="${DTYPE:-bf16}"
QUANTIZATION="${QUANTIZATION:-none}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

TASKS=(
  "sft|HuggingFaceTB/SmolVLM2-2.2B-Instruct"
  "sft|google/paligemma2-3b-mix-448"
  "sft|microsoft/Phi-4-multimodal-instruct"
  "sft|allenai/Molmo2-4B"
  "sft|llava-hf/llava-onevision-qwen2-7b-ov-hf"
  "sft|Qwen/Qwen2.5-VL-3B-Instruct"
  "rl|HuggingFaceTB/SmolVLM2-2.2B-Instruct"
  "rl|google/paligemma2-3b-mix-448"
  "rl|microsoft/Phi-4-multimodal-instruct"
  "rl|allenai/Molmo2-4B"
  "rl|llava-hf/llava-onevision-qwen2-7b-ov-hf"
  "rl|Qwen/Qwen2.5-VL-3B-Instruct"
)

task="${TASKS[$SLURM_ARRAY_TASK_ID]}"
phase="${task%%|*}"
model_name="${task#*|}"

status_dir="$OUTPUT_DIR/status"
mkdir -p "$status_dir"

"${PYTHON_BIN}" benchmarks/vlm_baselines/build_phase_splits.py --phase both --write-report >/dev/null

set +e
run_args=(
  benchmarks/vlm_baselines/run_baselines.py
  --phase "$phase" \
  --split "$SPLIT" \
  --model-name "$model_name" \
  --batch-size 1 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --quantization "$QUANTIZATION" \
  --device cuda:0 \
  --output-dir "$OUTPUT_DIR"
)
if [[ "$MAX_SAMPLES" != "0" ]]; then
  run_args+=(--max-samples "$MAX_SAMPLES")
fi
"${PYTHON_BIN}" "${run_args[@]}"
exit_code=$?
set -e

printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$exit_code" "$phase" "$model_name" >> "$status_dir/model_status.tsv"
exit "$exit_code"
