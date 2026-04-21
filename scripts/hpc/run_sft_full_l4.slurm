#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-full-l4
#SBATCH --partition=hpg-turin
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
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
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"

mkdir -p "$AGRI_VLM_DATA_ROOT" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_XET_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

python scripts/verify_environment.py
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc '
python -m torch.distributed.run \
  --nnodes "$SLURM_NNODES" \
  --node_rank "$SLURM_NODEID" \
  --nproc_per_node 2 \
  --rdzv_backend c10d \
  --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id "$SLURM_JOB_ID" \
  --max_restarts 0 \
  scripts/train/train_sft.py \
  --model-config configs/model/qwen_vlm_4b_l4.yaml \
  --train-config configs/train/sft_lora_full_l4_multigpu.yaml
'
