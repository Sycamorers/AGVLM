#!/bin/bash -l
#SBATCH --job-name=agri-vlm-eval-llama4-sft32
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd /blue/hmedeiros/qinruoyao/agvlm

module load conda
module load cuda/12.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

export AGRI_VLM_DATA_ROOT="$PWD/data"
export HF_HOME="$PWD/.cache/huggingface"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$PWD/.tmp"
export TRITON_CACHE_DIR="$TMPDIR/triton-eval-$SLURM_JOB_ID"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export AGRI_VLM_REQUIRED_MODEL_ACCESS="meta-llama/Llama-4-Scout-17B-16E-Instruct"
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-full-turin-16gpu-zero3}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/llama4_scout_17b_16e_turin_24g_lowres_infer.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval/sft_eval_llama4_max1_stratified512_verify32.yaml}"
PREDICTIONS_OUTPUT="${PREDICTIONS_OUTPUT:-outputs/eval/verification/llama4_scout_sft_eval512_verify32_predictions.jsonl}"

mkdir -p \
  "$AGRI_VLM_DATA_ROOT" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$HF_XET_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TMPDIR" \
  "$TRITON_CACHE_DIR" \
  outputs/eval/verification

python scripts/verify_environment.py
python scripts/eval/eval_local_holdout.py \
  --model-config "$MODEL_CONFIG" \
  --eval-config "$EVAL_CONFIG" \
  --prediction-mode model \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --predictions-output "$PREDICTIONS_OUTPUT"

rm -rf "$TRITON_CACHE_DIR"
