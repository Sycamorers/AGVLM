#!/bin/bash -l
#SBATCH --job-name=agri-sft-bench
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
export PYTHONPATH="$PWD/benchmarks/vlm_baselines${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TMPDIR="$PWD/benchmarks/vlm_baselines/tmp/${SLURM_JOB_ID:-local}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_RUNTIME_DIR="$TMPDIR/xdg_runtime"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TMPDIR/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR" "$XDG_RUNTIME_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" benchmarks/vlm_baselines/logs/slurm benchmarks/vlm_baselines/results
chmod 700 "$XDG_RUNTIME_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
MODEL_KEY="${MODEL_KEY:-}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-2}"
DTYPE="${DTYPE:-bf16}"
QUANTIZATION="${QUANTIZATION:-}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/vlm_baselines/results}"
SPLIT_DIR="${SPLIT_DIR:-benchmarks/vlm_baselines/splits}"
FORCE_SPLITS="${FORCE_SPLITS:-0}"
DRY_RUN="${DRY_RUN:-0}"

echo "git_commit=$(git rev-parse HEAD)"
echo "phase=sft split=${SPLIT} model_name=${MODEL_NAME} model_key=${MODEL_KEY} output_dir=${OUTPUT_DIR} split_dir=${SPLIT_DIR} max_samples=${MAX_SAMPLES} min_new_tokens=${MIN_NEW_TOKENS} quantization_override=${QUANTIZATION:-<model-config>}"
nvidia-smi || true

split_args=(
  benchmarks/vlm_baselines/build_phase_splits.py
  --phase sft
  --output-dir "$SPLIT_DIR"
  --write-report
)
if [[ "$FORCE_SPLITS" == "1" ]]; then
  split_args+=(--force)
fi
"${PYTHON_BIN}" "${split_args[@]}"
test -s "$SPLIT_DIR/sft_${SPLIT}_manifest.jsonl"

args=(
  benchmarks/vlm_baselines/run_baselines.py
  --phase sft
  --split "$SPLIT"
  --split-dir "$SPLIT_DIR"
  --batch-size 1
  --max-new-tokens "$MAX_NEW_TOKENS"
  --min-new-tokens "$MIN_NEW_TOKENS"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$QUANTIZATION" ]]; then
  args+=(--quantization "$QUANTIZATION")
fi
if [[ -n "$MODEL_KEY" ]]; then
  args+=(--model-key "$MODEL_KEY")
else
  args+=(--model-name "$MODEL_NAME")
fi
if [[ "$MAX_SAMPLES" != "0" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "${PYTHON_BIN}" "${args[@]}"
