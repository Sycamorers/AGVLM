#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-full-b200
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=23:00:00
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

python scripts/verify_environment.py
python scripts/launch_torchrun.py \
  --nproc-per-node 4 \
  scripts/train/train_sft.py -- \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_lora_full_b200_multigpu.yaml
