#!/bin/bash -l
#SBATCH --job-name=agri-rl-bench
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
export TMPDIR="${TMPDIR:-$PWD/benchmarks/vlm_baselines/tmp/${SLURM_JOB_ID:-local}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR" benchmarks/vlm_baselines/logs/slurm benchmarks/vlm_baselines/results

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
MODEL_KEY="${MODEL_KEY:-}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
DTYPE="${DTYPE:-bf16}"
QUANTIZATION="${QUANTIZATION:-none}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/vlm_baselines/results}"
DRY_RUN="${DRY_RUN:-0}"

echo "git_commit=$(git rev-parse HEAD)"
echo "phase=rl split=${SPLIT} model_name=${MODEL_NAME} model_key=${MODEL_KEY} output_dir=${OUTPUT_DIR} max_samples=${MAX_SAMPLES}"
nvidia-smi || true

"${PYTHON_BIN}" benchmarks/vlm_baselines/build_phase_splits.py --phase rl --write-report
test -s "benchmarks/vlm_baselines/splits/rl_${SPLIT}_manifest.jsonl"

args=(
  benchmarks/vlm_baselines/run_baselines.py
  --phase rl
  --split "$SPLIT"
  --batch-size 1
  --max-new-tokens "$MAX_NEW_TOKENS"
  --dtype "$DTYPE"
  --quantization "$QUANTIZATION"
  --device "$DEVICE"
  --output-dir "$OUTPUT_DIR"
)

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
