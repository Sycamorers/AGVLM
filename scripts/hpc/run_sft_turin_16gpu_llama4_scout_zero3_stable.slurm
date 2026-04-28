#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-llama4-16g-stable
#SBATCH --partition=hpg-turin
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=zijing.huang@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
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
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$PWD/.tmp"
export TRITON_CACHE_DIR="$TMPDIR/triton-$SLURM_JOB_ID"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export AGRI_VLM_REQUIRED_MODEL_ACCESS="meta-llama/Llama-4-Scout-17B-16E-Instruct"
export PYTHONPATH=src
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((28600 + SLURM_JOB_ID % 1000))"
RUN_DIR="$PWD/outputs/sft/llama4-scout-17b-16e-lora-full-turin-16gpu-zero3"

mkdir -p "$AGRI_VLM_DATA_ROOT" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_XET_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR"

if [[ -d "$RUN_DIR" ]]; then
  mv "$RUN_DIR" "$RUN_DIR.backup-$(date +%Y%m%d-%H%M%S)"
fi

python scripts/verify_environment.py
TRAIN_EXIT=0
set +e
srun --kill-on-bad-exit=1 --export=ALL,TORCH_CUDA_ARCH_LIST=8.9 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc '
disable_core_dumps() {
  ulimit -S -c 0 || true
  ulimit -H -c 0 || true
}
disable_core_dumps
export TORCH_CUDA_ARCH_LIST=8.9
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
launch=(python -m torch.distributed.run \
  --nnodes "$SLURM_NNODES" \
  --node_rank "$SLURM_NODEID" \
  --nproc_per_node 2 \
  --rdzv_backend c10d \
  --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id "$SLURM_JOB_ID" \
  --max_restarts 0 \
  scripts/train/train_sft.py \
  --model-config configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml \
  --train-config configs/train/sft_lora_turin_16gpu_llama4_scout_zero3_stable.yaml)
if command -v prlimit >/dev/null 2>&1; then
  prlimit --core=0:0 -- "${launch[@]}"
else
  "${launch[@]}"
fi
'
TRAIN_EXIT=$?
set -e

if [[ -f "$RUN_DIR/metrics/train_metrics.jsonl" || -f "$RUN_DIR/metrics.jsonl" ]]; then
  python scripts/artifacts/export_training_artifacts.py --run-dir "$RUN_DIR" || true
fi

rm -rf "$TRITON_CACHE_DIR"

echo "TensorBoard: tensorboard --logdir $RUN_DIR/tensorboard --host 0.0.0.0 --port 6007"
exit "$TRAIN_EXIT"
