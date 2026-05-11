#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-phi4rv-b200x4
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=56
#SBATCH --mem=650G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

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
export HUGGINGFACE_HUB_CACHE="$HF_HOME/transformers"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
export TMPDIR="$PWD/.tmp"
export TRITON_CACHE_DIR="$TMPDIR/triton-phi4rv-b200x4-max3-$SLURM_JOB_ID"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export AGRI_VLM_REQUIRED_MODEL_ACCESS="microsoft/Phi-4-reasoning-vision-15B"
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="10.0"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL,NET}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_FR_BUFFER_SIZE=1048576
export TORCH_NCCL_TRACE_CPP_STACK=1
export TORCH_NCCL_DESYNC_DEBUG=1
export AGRI_VLM_DISTRIBUTED_DEBUG="${AGRI_VLM_DISTRIBUTED_DEBUG:-1}"
export AGRI_VLM_DISTRIBUTED_DEBUG_MAX_STEPS="${AGRI_VLM_DISTRIBUTED_DEBUG_MAX_STEPS:-4}"
export SLURM_NNODES="${SLURM_NNODES:-1}"
export SLURM_NODEID="${SLURM_NODEID:-0}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((28600 + SLURM_JOB_ID % 1000))}"

DATA_CONFIG="${DATA_CONFIG:-configs/data/sft_train_eval_phi4_max3.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/phi4_reasoning_vision_15b_b200.yaml}"
PREFLIGHT_CONFIG="${PREFLIGHT_CONFIG:-configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_full_max3_preflight.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_full_max3.yaml}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6008}"
B200_GPUS_PER_NODE="${B200_GPUS_PER_NODE:-4}"
export DATA_CONFIG MODEL_CONFIG PREFLIGHT_CONFIG TRAIN_CONFIG TENSORBOARD_PORT B200_GPUS_PER_NODE

TENSORBOARD_PID=""

cleanup() {
  if [[ -n "${TENSORBOARD_PID:-}" ]] && kill -0 "$TENSORBOARD_PID" 2>/dev/null; then
    kill "$TENSORBOARD_PID" 2>/dev/null || true
    wait "$TENSORBOARD_PID" 2>/dev/null || true
  fi
  rm -rf "$TRITON_CACHE_DIR"
}
trap cleanup EXIT

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

fail_fast() {
  echo "ERROR: $1" >&2
  exit 2
}

run_training() {
  local train_config="$1"
  srun \
    --kill-on-bad-exit=1 \
    --export=ALL,TORCH_CUDA_ARCH_LIST=10.0 \
    --ntasks="$SLURM_NNODES" \
    --ntasks-per-node=1 \
    bash -lc '
disable_core_dumps() {
  ulimit -S -c 0 || true
  ulimit -H -c 0 || true
}
disable_core_dumps
export TORCH_CUDA_ARCH_LIST=10.0
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL,NET}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_FR_BUFFER_SIZE=1048576
export TORCH_NCCL_TRACE_CPP_STACK=1
export TORCH_NCCL_DESYNC_DEBUG=1
export AGRI_VLM_DISTRIBUTED_DEBUG="${AGRI_VLM_DISTRIBUTED_DEBUG:-1}"
export AGRI_VLM_DISTRIBUTED_DEBUG_MAX_STEPS="${AGRI_VLM_DISTRIBUTED_DEBUG_MAX_STEPS:-4}"
export SLURM_NODEID="${SLURM_NODEID:-0}"
launch=(python -m torch.distributed.run \
  --nnodes "$SLURM_NNODES" \
  --node_rank "$SLURM_NODEID" \
  --nproc_per_node 4 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --max_restarts 0 \
  scripts/train/train_sft.py \
  --model-config "$MODEL_CONFIG" \
  --train-config "'"$train_config"'")
if command -v prlimit >/dev/null 2>&1; then
  prlimit --core=0:0 -- "${launch[@]}"
else
  "${launch[@]}"
fi
'
}

start_tensorboard() {
  local run_dir="$1"
  mkdir -p "$run_dir/tensorboard"
  tensorboard \
    --logdir "$run_dir/tensorboard" \
    --host 0.0.0.0 \
    --port "$TENSORBOARD_PORT" \
    --reload_interval 30 \
    > "$run_dir/tensorboard-server.log" 2>&1 &
  TENSORBOARD_PID=$!
  sleep 5
  if ! kill -0 "$TENSORBOARD_PID" 2>/dev/null; then
    echo "TensorBoard failed to start. Log follows:"
    sed -n '1,120p' "$run_dir/tensorboard-server.log" || true
    exit 3
  fi
  echo "TensorBoard running inside B200 training job on $(hostname -f):$TENSORBOARD_PORT"
  echo "TensorBoard log file: $run_dir/tensorboard-server.log"
}

mkdir -p \
  "$AGRI_VLM_DATA_ROOT" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$HF_XET_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TMPDIR" \
  "$TRITON_CACHE_DIR" \
  logs/slurm

[[ -f "$DATA_CONFIG" ]] || fail_fast "data config missing: $DATA_CONFIG"
[[ -f "$MODEL_CONFIG" ]] || fail_fast "model config missing: $MODEL_CONFIG"
[[ -f "$PREFLIGHT_CONFIG" ]] || fail_fast "preflight config missing: $PREFLIGHT_CONFIG"
[[ -f "$TRAIN_CONFIG" ]] || fail_fast "train config missing: $TRAIN_CONFIG"

PREFLIGHT_RUN_DIR="$(config_value "$PREFLIGHT_CONFIG" output_dir)"
RUN_DIR="$(config_value "$TRAIN_CONFIG" output_dir)"
CHECKPOINT_RUN_DIR="$(config_value "$TRAIN_CONFIG" checkpoint_output_dir)"

[[ "$PREFLIGHT_RUN_DIR" == outputs/preflight/* ]] || fail_fast "preflight output_dir must be under outputs/preflight: $PREFLIGHT_RUN_DIR"
[[ "$RUN_DIR" == outputs/sft/* ]] || fail_fast "full output_dir must be under outputs/sft: $RUN_DIR"
[[ "$RUN_DIR" != "$PREFLIGHT_RUN_DIR" ]] || fail_fast "preflight and full output directories must differ"
[[ -n "$CHECKPOINT_RUN_DIR" ]] || fail_fast "checkpoint_output_dir is empty in $TRAIN_CONFIG"

echo "date=$(date -Is)"
echo "hostname=$(hostname -f)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

python scripts/verify_environment.py
python scripts/data/build_sft_train_eval_manifests.py --config "$DATA_CONFIG"

echo "Running B200 Phi-4 reasoning vision SFT preflight: $PREFLIGHT_CONFIG"
set +e
run_training "$PREFLIGHT_CONFIG"
PREFLIGHT_EXIT=$?
set -e
if [[ "$PREFLIGHT_EXIT" -ne 0 ]]; then
  echo "B200 preflight failed with exit $PREFLIGHT_EXIT. Full training will not start."
  exit "$PREFLIGHT_EXIT"
fi
echo "B200 preflight succeeded."

mkdir -p "$RUN_DIR" "$CHECKPOINT_RUN_DIR"
echo "Full train config: $TRAIN_CONFIG"
echo "Full run directory: $RUN_DIR"
echo "Full checkpoint directory: $CHECKPOINT_RUN_DIR"

start_tensorboard "$RUN_DIR"

METRICS_CANDIDATES="$RUN_DIR/metrics/train_metrics.jsonl,$RUN_DIR/metrics.jsonl"
echo "B200_FULL_TRAINING_STARTED=1"
echo "B200_RUN_DIR=$RUN_DIR"
echo "B200_CHECKPOINT_DIR=$CHECKPOINT_RUN_DIR"
echo "B200_METRICS_CANDIDATES=$METRICS_CANDIDATES"

TRAIN_EXIT=0
set +e
run_training "$TRAIN_CONFIG"
TRAIN_EXIT=$?
set -e

if [[ -f "$RUN_DIR/metrics/train_metrics.jsonl" || -f "$RUN_DIR/metrics.jsonl" ]]; then
  python scripts/artifacts/export_training_artifacts.py --run-dir "$RUN_DIR" || true
fi

echo "TensorBoard was running inside this B200 training job on port $TENSORBOARD_PORT"
exit "$TRAIN_EXIT"
