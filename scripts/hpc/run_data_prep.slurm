#!/usr/bin/env bash
#SBATCH --job-name=agri-vlm-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
##SBATCH --partition=<set-your-partition>

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export ENV_NAME="${ENV_NAME:-agri-vlm-v1}"
export DOWNLOAD_MODE="${DOWNLOAD_MODE:-partial}"
export SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"

bash scripts/hpc/run_data_prep.sh
