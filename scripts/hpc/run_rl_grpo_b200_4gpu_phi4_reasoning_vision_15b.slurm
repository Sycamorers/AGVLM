#!/bin/bash -l
#SBATCH --job-name=agri-vlm-rl-phi4rv-b200x4
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --constraint=b200
#SBATCH --mem=360G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd /blue/hmedeiros/qinruoyao/agvlm

module load conda
module load cuda/12.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

export PYTHONPATH=src
export AGRI_VLM_DATA_ROOT="${AGRI_VLM_DATA_ROOT:-$PWD/data}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TRANSFORMERS_CACHE}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export TMPDIR="${TMPDIR:-$PWD/.tmp/rl-grpo-$SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/phi4_reasoning_vision_15b_turin_24g.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_smoke_after_sft.yaml}"
MASTER_PORT="${MASTER_PORT:-29541}"
export MODEL_CONFIG TRAIN_CONFIG MASTER_PORT

mkdir -p logs/slurm "$AGRI_VLM_DATA_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"

config_value() {
  python - "$1" "$2" <<'PY'
import sys
import yaml
from pathlib import Path

payload = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
value = payload.get(sys.argv[2])
print("" if value is None else value)
PY
}

model_value() {
  python - "$1" "$2" <<'PY'
import sys
import yaml
from pathlib import Path

payload = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
value = payload.get(sys.argv[2])
print("" if value is None else value)
PY
}

fail_fast() {
  echo "ERROR: $1" >&2
  exit 2
}

[[ -f "$MODEL_CONFIG" ]] || fail_fast "model config missing: $MODEL_CONFIG"
[[ -f "$TRAIN_CONFIG" ]] || fail_fast "train config missing: $TRAIN_CONFIG"

MANIFEST_PATH="$(config_value "$TRAIN_CONFIG" manifest_path)"
SFT_CHECKPOINT_PATH="$(config_value "$TRAIN_CONFIG" sft_checkpoint_path)"
BASE_MODEL_NAME="$(model_value "$MODEL_CONFIG" model_name_or_path)"

[[ -n "$MANIFEST_PATH" ]] || fail_fast "manifest_path is empty in $TRAIN_CONFIG"
[[ -f "$MANIFEST_PATH" ]] || fail_fast "RL manifest missing: $MANIFEST_PATH"
[[ -n "$SFT_CHECKPOINT_PATH" ]] || fail_fast "sft_checkpoint_path is empty in $TRAIN_CONFIG"
if [[ "$SFT_CHECKPOINT_PATH" == *"<"* ]] || [[ "$SFT_CHECKPOINT_PATH" == *">"* ]] || [[ "$SFT_CHECKPOINT_PATH" == *"FINAL_SFT_CHECKPOINT_OR_ADAPTER"* ]]; then
  fail_fast "sft_checkpoint_path still contains a placeholder: $SFT_CHECKPOINT_PATH"
fi
if [[ "$SFT_CHECKPOINT_PATH" == "$BASE_MODEL_NAME" ]] || [[ "$SFT_CHECKPOINT_PATH" == "microsoft/Phi-4-reasoning-vision-15B" ]]; then
  fail_fast "RLFT must start from a completed SFT checkpoint, not the raw/base model: $SFT_CHECKPOINT_PATH"
fi
[[ -e "$SFT_CHECKPOINT_PATH" ]] || fail_fast "SFT checkpoint path does not exist: $SFT_CHECKPOINT_PATH"

echo "date=$(date -Is)"
echo "hostname=$(hostname -f)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
python - <<'PY'
import sys
try:
    import torch
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
except Exception as exc:
    torch_version = "unavailable: %s" % exc
    cuda_available = False
    cuda_device_count = 0
print("python=%s" % sys.executable)
print("torch=%s" % torch_version)
print("cuda_available=%s" % cuda_available)
print("cuda_device_count=%s" % cuda_device_count)
PY
echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"
echo "resolved_manifest_path=$MANIFEST_PATH"
echo "resolved_sft_checkpoint_path=$SFT_CHECKPOINT_PATH"

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port "$MASTER_PORT" \
  scripts/train/train_rl_grpo.py \
  --model-config "$MODEL_CONFIG" \
  --train-config "$TRAIN_CONFIG"
