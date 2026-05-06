#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-full-max3-b200
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=512G
#SBATCH --time=1-14:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zijing.huang@ufl.edu

set -euo pipefail

cd /blue/hmedeiros/qinruoyao/agvlm

module load conda
module load cuda/12.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

disable_core_dumps() {
  ulimit -S -c 0 || true
  ulimit -H -c 0 || true
}
disable_core_dumps

export AGRI_VLM_DATA_ROOT="$PWD/data"
export HF_HOME="$PWD/.cache/huggingface"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$PWD/.tmp"
export TRITON_CACHE_DIR="$TMPDIR/triton-full-max3-b200-$SLURM_JOB_ID"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export AGRI_VLM_REQUIRED_MODEL_ACCESS="meta-llama/Llama-4-Scout-17B-16E-Instruct"
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

DATA_CONFIG="${DATA_CONFIG:-configs/data/sft_train_eval_llama4_max3.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced.yaml}"
RUN_DIR="$(python - "$TRAIN_CONFIG" <<'PY'
import sys
import yaml
from pathlib import Path

payload = yaml.safe_load(Path(sys.argv[1]).read_text())
print(payload["output_dir"])
PY
)"
UPSTREAM_CHECKPOINT_DIR="/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft"

mkdir -p \
  "$AGRI_VLM_DATA_ROOT" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$HF_XET_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TMPDIR" \
  "$TRITON_CACHE_DIR" \
  "$RUN_DIR" \
  logs/slurm

python scripts/verify_environment.py

if [[ ! -s "$UPSTREAM_CHECKPOINT_DIR/adapter_config.json" || ! -s "$UPSTREAM_CHECKPOINT_DIR/adapter_model.safetensors" ]]; then
  echo "Missing usable upstream adapter at $UPSTREAM_CHECKPOINT_DIR"
  exit 2
fi
echo "Using upstream SFT adapter: $UPSTREAM_CHECKPOINT_DIR"

python scripts/data/build_sft_train_eval_manifests.py --config "$DATA_CONFIG"

TRAIN_EXIT=0
set +e
python scripts/launch_torchrun.py \
  --nproc-per-node 4 \
  scripts/train/train_sft.py -- \
  --model-config configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml \
  --train-config "$TRAIN_CONFIG"
TRAIN_EXIT=$?
set -e

if [[ -f "$RUN_DIR/metrics/train_metrics.jsonl" || -f "$RUN_DIR/metrics.jsonl" ]]; then
  python scripts/artifacts/export_training_artifacts.py --run-dir "$RUN_DIR" || true
fi

rm -rf "$TRITON_CACHE_DIR"

echo "TensorBoard: tensorboard --logdir $RUN_DIR/tensorboard --host 0.0.0.0 --port 6013"
exit "$TRAIN_EXIT"
