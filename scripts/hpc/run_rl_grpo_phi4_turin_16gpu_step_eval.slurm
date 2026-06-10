#!/bin/bash -l
#SBATCH --job-name=agri-vlm-rl-turin-step-eval
#SBATCH --partition=hpg-turin
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=80G
#SBATCH --time=01:00:00
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
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/transformers}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export TMPDIR="${TMPDIR:-$PWD/.tmp/rl-turin-step-eval-$SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export TORCH_CUDA_ARCH_LIST=8.9
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((28700 + SLURM_JOB_ID % 1000))}"

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/phi4_reasoning_vision_15b_turin_24g_4bit_rl.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/rl_grpo_phi4_turin_16gpu_step_eval_4bit_from_sft1700.yaml}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-2}"
export MODEL_CONFIG TRAIN_CONFIG EVAL_MAX_EXAMPLES

mkdir -p logs/slurm "$AGRI_VLM_DATA_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR"

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

[[ -f "$MODEL_CONFIG" ]] || fail_fast "model config missing: $MODEL_CONFIG"
[[ -f "$TRAIN_CONFIG" ]] || fail_fast "train config missing: $TRAIN_CONFIG"

MANIFEST_PATH="$(config_value "$TRAIN_CONFIG" manifest_path)"
EVAL_MANIFEST_PATH="$(config_value "$TRAIN_CONFIG" eval_manifest_path)"
SFT_CHECKPOINT_PATH="$(config_value "$TRAIN_CONFIG" sft_checkpoint_path)"
CHECKPOINT_OUTPUT_DIR="$(config_value "$TRAIN_CONFIG" checkpoint_output_dir)"
export MANIFEST_PATH EVAL_MANIFEST_PATH SFT_CHECKPOINT_PATH CHECKPOINT_OUTPUT_DIR

[[ -f "$MANIFEST_PATH" ]] || fail_fast "RL manifest missing: $MANIFEST_PATH"
[[ -f "$EVAL_MANIFEST_PATH" ]] || fail_fast "RL eval manifest missing: $EVAL_MANIFEST_PATH"
[[ -d "$SFT_CHECKPOINT_PATH" ]] || fail_fast "SFT checkpoint path missing: $SFT_CHECKPOINT_PATH"

python - "$MODEL_CONFIG" "$TRAIN_CONFIG" <<'PY'
import sys
from pathlib import Path
from agri_vlm.schemas.config_schema import ModelConfigSchema, RLTrainConfigSchema, load_config
from agri_vlm.training.rl_trainer import validate_rl_sft_checkpoint_path

model = load_config(Path(sys.argv[1]), ModelConfigSchema)
train = load_config(Path(sys.argv[2]), RLTrainConfigSchema)
validate_rl_sft_checkpoint_path(model_config=model, train_config=train)
PY

echo "date=$(date -Is)"
echo "hostname=$(hostname -f)"
echo "git_commit=$(git rev-parse HEAD)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"
echo "MANIFEST_PATH=$MANIFEST_PATH"
echo "EVAL_MANIFEST_PATH=$EVAL_MANIFEST_PATH"
echo "SFT_CHECKPOINT_PATH=$SFT_CHECKPOINT_PATH"
echo "CHECKPOINT_OUTPUT_DIR=$CHECKPOINT_OUTPUT_DIR"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

srun --kill-on-bad-exit=1 --export=ALL,TORCH_CUDA_ARCH_LIST=8.9 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc '
python -m torch.distributed.run \
  --nnodes "$SLURM_NNODES" \
  --node_rank "$SLURM_NODEID" \
  --nproc_per_node 2 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --max_restarts 0 \
  scripts/train/train_rl_grpo.py \
  --model-config "$MODEL_CONFIG" \
  --train-config "$TRAIN_CONFIG"
'

if [[ ! -d "$CHECKPOINT_OUTPUT_DIR/checkpoint-1" && ! -f "$CHECKPOINT_OUTPUT_DIR/adapter_model.safetensors" ]]; then
  fail_fast "RL test did not produce checkpoint artifacts under $CHECKPOINT_OUTPUT_DIR"
fi

srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --export=ALL bash -lc '
PYTHONPATH=src python scripts/eval/eval_rl_checkpoint.py \
  --model-config "$MODEL_CONFIG" \
  --manifest-path "$EVAL_MANIFEST_PATH" \
  --checkpoint-path "$CHECKPOINT_OUTPUT_DIR" \
  --prediction-mode model \
  --max-examples "$EVAL_MAX_EXAMPLES" \
  --batch-size 1 \
  --max-new-tokens 96 \
  --metrics-output reports/rl_turin_step_eval_metrics.json \
  --samples-output reports/rl_turin_step_eval_samples.jsonl \
  --report-output reports/rl_turin_step_eval_report.md
'

echo "rl_turin_step_eval_completed=1"
