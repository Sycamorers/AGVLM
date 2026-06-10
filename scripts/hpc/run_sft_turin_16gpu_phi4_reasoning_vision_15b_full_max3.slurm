#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-phi4rv-16g
#SBATCH --partition=hpg-turin
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=zijing.huang@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${AGRI_VLM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"

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
export TRITON_CACHE_DIR="$TMPDIR/triton-phi4rvx16-max3-$SLURM_JOB_ID"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export AGRI_VLM_REQUIRED_MODEL_ACCESS="microsoft/Phi-4-reasoning-vision-15B"
export PYTHONPATH=src
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="8.9"
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
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((28600 + SLURM_JOB_ID % 1000))"

DATA_CONFIG="${DATA_CONFIG:-configs/data/sft_train_eval_phi4_max3.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/phi4_reasoning_vision_15b_turin_24g.yaml}"
BASE_TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_full_max3.yaml}"
BASE_PREFLIGHT_CONFIG="${PREFLIGHT_CONFIG:-configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_full_max3_preflight.yaml}"
PHI4_BATCH_CANDIDATES="${PHI4_BATCH_CANDIDATES:-1}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6007}"
export DATA_CONFIG MODEL_CONFIG BASE_TRAIN_CONFIG BASE_PREFLIGHT_CONFIG TENSORBOARD_PORT
GENERATED_CONFIG_DIR="$TMPDIR/generated-configs/phi4rv-$SLURM_JOB_ID"
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

payload = yaml.safe_load(Path(sys.argv[1]).read_text())
print(payload.get(sys.argv[2]) or "")
PY
}

write_batch_config() {
  local source_config="$1"
  local target_config="$2"
  local batch_size="$3"
  local run_kind="$4"
  python - "$source_config" "$target_config" "$batch_size" "$run_kind" <<'PY'
import sys
import yaml
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
batch_size = int(sys.argv[3])
run_kind = sys.argv[4]
payload = yaml.safe_load(source.read_text())
payload["per_device_train_batch_size"] = batch_size
payload["per_device_eval_batch_size"] = batch_size
suffix = "batch%s" % batch_size
def append_suffix(value):
    text = str(value)
    if text.endswith("-%s" % suffix):
        return text
    return "%s-%s" % (text, suffix)
if run_kind == "preflight":
    payload["output_dir"] = append_suffix(payload["output_dir"])
    payload["checkpoint_output_dir"] = append_suffix(payload["checkpoint_output_dir"])
    payload["save_strategy"] = "no"
    payload["save_final_model"] = False
    payload["eval_manifest_path"] = None
else:
    payload["output_dir"] = append_suffix(payload["output_dir"])
    payload["checkpoint_output_dir"] = append_suffix(payload["checkpoint_output_dir"])
    payload["save_final_model"] = True
if run_kind == "full" and (
    str(payload["output_dir"]).startswith("outputs/smoke")
    or "/outputs/smoke/" in str(payload["checkpoint_output_dir"])
):
    raise ValueError("Full training output paths must not use outputs/smoke.")
payload["run_name"] = "%s-%s" % (payload["run_name"], suffix)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
}

run_training() {
  local train_config="$1"
  srun --kill-on-bad-exit=1 --export=ALL,TORCH_CUDA_ARCH_LIST=8.9 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc '
disable_core_dumps() {
  ulimit -S -c 0 || true
  ulimit -H -c 0 || true
}
disable_core_dumps
export TORCH_CUDA_ARCH_LIST=8.9
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
launch=(python -m torch.distributed.run \
  --nnodes "$SLURM_NNODES" \
  --node_rank "$SLURM_NODEID" \
  --nproc_per_node 2 \
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
  echo "TensorBoard running inside training job on $(hostname -f):$TENSORBOARD_PORT"
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
  "$GENERATED_CONFIG_DIR" \
  logs/slurm

python scripts/verify_environment.py
python scripts/data/build_sft_train_eval_manifests.py --config "$DATA_CONFIG"

SELECTED_BATCH=""
SELECTED_PREFLIGHT_CONFIG=""
for BATCH_SIZE in $PHI4_BATCH_CANDIDATES; do
  PREFLIGHT_CONFIG_PATH="$GENERATED_CONFIG_DIR/preflight-batch${BATCH_SIZE}.yaml"
  write_batch_config "$BASE_PREFLIGHT_CONFIG" "$PREFLIGHT_CONFIG_PATH" "$BATCH_SIZE" preflight
  echo "Testing Phi-4 reasoning vision per-device batch size $BATCH_SIZE with 2-step preflight."
  set +e
  run_training "$PREFLIGHT_CONFIG_PATH"
  PREFLIGHT_EXIT=$?
  set -e
  if [[ "$PREFLIGHT_EXIT" -eq 0 ]]; then
    SELECTED_BATCH="$BATCH_SIZE"
    SELECTED_PREFLIGHT_CONFIG="$PREFLIGHT_CONFIG_PATH"
    echo "Selected Phi-4 reasoning vision per-device batch size $SELECTED_BATCH."
    break
  fi
  echo "Phi-4 reasoning vision per-device batch size $BATCH_SIZE failed preflight with exit $PREFLIGHT_EXIT."
done

if [[ -z "$SELECTED_BATCH" ]]; then
  echo "No Phi-4 reasoning vision batch-size candidate passed preflight: $PHI4_BATCH_CANDIDATES"
  exit 2
fi

TRAIN_CONFIG_PATH="$GENERATED_CONFIG_DIR/full-batch${SELECTED_BATCH}.yaml"
write_batch_config "$BASE_TRAIN_CONFIG" "$TRAIN_CONFIG_PATH" "$SELECTED_BATCH" full
RUN_DIR="$(config_value "$TRAIN_CONFIG_PATH" output_dir)"
CHECKPOINT_RUN_DIR="$(config_value "$TRAIN_CONFIG_PATH" checkpoint_output_dir)"
mkdir -p "$RUN_DIR" "$CHECKPOINT_RUN_DIR"
echo "Preflight config: $SELECTED_PREFLIGHT_CONFIG"
echo "Full train config: $TRAIN_CONFIG_PATH"
echo "Full run directory: $RUN_DIR"
echo "Full checkpoint directory: $CHECKPOINT_RUN_DIR"

start_tensorboard "$RUN_DIR"

TRAIN_EXIT=0
set +e
run_training "$TRAIN_CONFIG_PATH"
TRAIN_EXIT=$?
set -e

if [[ -f "$RUN_DIR/metrics/train_metrics.jsonl" || -f "$RUN_DIR/metrics.jsonl" ]]; then
  python scripts/artifacts/export_training_artifacts.py --run-dir "$RUN_DIR" || true
fi

echo "TensorBoard was running inside this training job on port $TENSORBOARD_PORT"
exit "$TRAIN_EXIT"
