#!/bin/bash -l
#SBATCH --job-name=agri-vlm-rl-turin8-smoke
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd /blue/hmedeiros/qinruoyao/agvlm

module load conda
module load cuda || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

export PYTHONPATH=src
export AGRI_VLM_DATA_ROOT="${AGRI_VLM_DATA_ROOT:-$PWD/data}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TRANSFORMERS_CACHE}"
export TMPDIR="${TMPDIR:-$PWD/.tmp/rl-grpo-turin8-smoke-$SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export AGRI_VLM_REWARD_LOG_EVERY="${AGRI_VLM_REWARD_LOG_EVERY:-1}"
export AGRI_VLM_REWARD_DIAGNOSTICS_JSONL="${AGRI_VLM_REWARD_DIAGNOSTICS_JSONL:-outputs/rl/smoke/grpo-phi4-turin8-tiny/reward_diagnostics.jsonl}"

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/phi4_reasoning_vision_15b_turin_24g_4bit_rl.yaml}"
TRAIN_CONFIG_BASE="${TRAIN_CONFIG:-configs/train/rl_grpo_phi4_turin8_tiny_smoke.yaml}"
MASTER_PORT="${MASTER_PORT:-29543}"
mkdir -p logs/slurm "$AGRI_VLM_DATA_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR" "$(dirname "$AGRI_VLM_REWARD_DIAGNOSTICS_JSONL")"

fail_fast() {
  echo "ERROR: $1" >&2
  exit 2
}

[[ -f "$MODEL_CONFIG" ]] || fail_fast "model config missing: $MODEL_CONFIG"
[[ -f "$TRAIN_CONFIG_BASE" ]] || fail_fast "train config missing: $TRAIN_CONFIG_BASE"

TRAIN_CONFIG="$TRAIN_CONFIG_BASE"
if [[ -n "${SFT_CHECKPOINT_PATH:-}" ]]; then
  TRAIN_CONFIG="$TMPDIR/rl_grpo_phi4_turin8_tiny_smoke.resolved.yaml"
  python - "$TRAIN_CONFIG_BASE" "$TRAIN_CONFIG" "$SFT_CHECKPOINT_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

source, target, checkpoint = sys.argv[1:4]
payload = yaml.safe_load(Path(source).read_text()) or {}
payload["sft_checkpoint_path"] = checkpoint
Path(target).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
fi

python - "$TRAIN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

payload = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
checkpoint = str(payload.get("sft_checkpoint_path") or "")
manifest = str(payload.get("manifest_path") or "")
if not manifest or not Path(manifest).exists():
    raise SystemExit("RL manifest missing: %s" % manifest)
if not checkpoint or "<" in checkpoint or ">" in checkpoint:
    raise SystemExit("Set SFT_CHECKPOINT_PATH or edit sft_checkpoint_path before submitting this smoke test.")
if not Path(checkpoint).exists():
    raise SystemExit("SFT checkpoint path does not exist: %s" % checkpoint)
print("turin8_smoke_manifest=%s" % manifest)
print("turin8_smoke_sft_checkpoint=%s" % checkpoint)
print("turin8_smoke_max_steps=%s smoke_max_samples=%s" % (payload.get("max_steps"), payload.get("smoke_max_samples")))
PY

echo "date=$(date -Is)"
echo "hostname=$(hostname -f)"
echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"
echo "AGRI_VLM_REWARD_DIAGNOSTICS_JSONL=$AGRI_VLM_REWARD_DIAGNOSTICS_JSONL"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

PYTHONPATH=src python scripts/validate_rl_manifest.py \
  --manifest "$(python - "$TRAIN_CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
print((yaml.safe_load(Path(sys.argv[1]).read_text()) or {})["manifest_path"])
PY
)"

PYTHONPATH=src python scripts/score_rl_manifest.py \
  --manifest "$(python - "$TRAIN_CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
print((yaml.safe_load(Path(sys.argv[1]).read_text()) or {})["manifest_path"])
PY
)" \
  --output outputs/rl/smoke/grpo-phi4-turin8-tiny/reward_score_preflight.jsonl \
  --summary-output outputs/rl/smoke/grpo-phi4-turin8-tiny/reward_score_preflight.summary.json \
  --max-samples 8

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port "$MASTER_PORT" \
  scripts/train/train_rl_grpo.py \
  --model-config "$MODEL_CONFIG" \
  --train-config "$TRAIN_CONFIG"
