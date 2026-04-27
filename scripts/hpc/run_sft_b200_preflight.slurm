#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-b200-preflight
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --mail-user=zijing.huang@ufl.edu
#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT
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
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$PWD/.tmp"
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_DIR="$PWD/outputs/smoke/sft-qwen3-vl-4b-b200-preflight"

mkdir -p "$AGRI_VLM_DATA_ROOT" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_XET_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

python scripts/verify_environment.py
python - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "CUDA is not available on the allocated B200 node."
names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
assert len(names) >= 2, "Expected at least 2 visible GPUs, found: %s" % names
assert all("B200" in name for name in names[:2]), "Expected NVIDIA B200 GPUs, found: %s" % names
assert torch.cuda.is_bf16_supported(), "BF16 is not supported by the allocated B200 runtime."
print("b200_cuda_validation=%s" % json.dumps({"gpu_names": names[:2], "bf16_supported": True}, sort_keys=True))
PY

TRAIN_EXIT=0
set +e
python scripts/launch_torchrun.py \
  --nproc-per-node 2 \
  --master-port "$((28600 + SLURM_JOB_ID % 1000))" \
  scripts/train/train_sft.py -- \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_lora_full_b200_preflight.yaml
TRAIN_EXIT=$?
set -e

if [[ -f "$RUN_DIR/metrics/train_metrics.jsonl" || -f "$RUN_DIR/metrics.jsonl" ]]; then
  python scripts/artifacts/export_training_artifacts.py --run-dir "$RUN_DIR" || true
fi

echo "TensorBoard: tensorboard --logdir $RUN_DIR/tensorboard --host 0.0.0.0 --port 6006"
exit "$TRAIN_EXIT"
