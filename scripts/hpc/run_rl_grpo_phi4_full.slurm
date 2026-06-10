#!/bin/bash -l
#SBATCH --job-name=agri-vlm-rl-phi4-full
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

REPO_ROOT="${AGRI_VLM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"

export MODEL_CONFIG="${MODEL_CONFIG:-configs/model/phi4_reasoning_vision_15b_turin_24g.yaml}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/rl_grpo_phi4_full.yaml}"
export MASTER_PORT="${MASTER_PORT:-29541}"

exec bash scripts/hpc/run_rl_grpo_b200_4gpu_phi4_reasoning_vision_15b.slurm
