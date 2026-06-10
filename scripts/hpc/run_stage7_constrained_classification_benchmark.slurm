#!/bin/bash -l
#SBATCH --job-name=agvlm-stage7-constrained
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${AGRI_VLM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"

module load conda
module load cuda/12.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

export PYTHONPATH=src
export AGRI_VLM_DATA_ROOT="${AGRI_VLM_DATA_ROOT:-$PWD/data}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export TMPDIR="${AGRI_VLM_TMPDIR:-$PWD/.tmp/stage7-constrained-${SLURM_JOB_ID:-manual}}"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TMPDIR/xdg-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export AGRI_VLM_CLASSIFICATION_PROMPT_FORMAT="${AGRI_VLM_CLASSIFICATION_PROMPT_FORMAT:-label_only}"

RUN_TAG="${RUN_TAG:-agvlm_stage7_label_only_classification_constrained_benchmark_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/vlm_baselines/results/$RUN_TAG}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

mkdir -p logs/slurm "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE" "$TMPDIR" "$OUTPUT_DIR"

echo "date=$(date -Is)"
echo "hostname=$(hostname -f)"
echo "git_commit=$(git rev-parse HEAD)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "AGRI_VLM_CLASSIFICATION_PROMPT_FORMAT=$AGRI_VLM_CLASSIFICATION_PROMPT_FORMAT"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true
fi

python benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split test \
  --split-dir benchmarks/vlm_baselines/splits_stage5_datafix \
  --batch-size 1 \
  --max-new-tokens 0 \
  --min-new-tokens 0 \
  --dtype bf16 \
  --quantization 4bit \
  --device cuda:0 \
  --output-dir "$OUTPUT_DIR" \
  --model-key agvlm_phi4_sft_stage7_label_only_classification_b200_candidate \
  --classification-decode-mode constrained \
  --max-samples "$MAX_SAMPLES"

echo "finished=$(date -Is)"
echo "output_dir=$OUTPUT_DIR"
