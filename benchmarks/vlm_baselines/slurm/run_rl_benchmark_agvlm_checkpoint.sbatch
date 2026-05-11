#!/bin/bash -l
#SBATCH --job-name=agri-rl-ckpt-bench
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=benchmarks/vlm_baselines/logs/slurm/%x-%j.out
#SBATCH --error=benchmarks/vlm_baselines/logs/slurm/%x-%j.err

set -euo pipefail

export MODEL_KEY="${MODEL_KEY:-agvlm_phi4_rl_completed}"
export MODEL_NAME=""
export OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/vlm_baselines/results}"

bash benchmarks/vlm_baselines/slurm/run_rl_benchmark_24gb.sbatch
