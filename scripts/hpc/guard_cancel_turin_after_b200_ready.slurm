#!/bin/bash -l
#SBATCH --job-name=agri-vlm-guard-b200-ready
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd /blue/hmedeiros/qinruoyao/agvlm

POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-60}"
MAX_GUARD_SECONDS="${MAX_GUARD_SECONDS:-21600}"
NOT_READY_EXIT_CODE="${NOT_READY_EXIT_CODE:-10}"
export NOT_READY_EXIT_CODE

mkdir -p logs/slurm

echo "date=$(date -Is)"
echo "guard_job_id=${SLURM_JOB_ID:-}"
echo "B200_JOB_ID=${B200_JOB_ID:-}"
echo "TURIN_JOB_ID=${TURIN_JOB_ID:-}"
echo "B200_RUN_DIR=${B200_RUN_DIR:-}"
echo "POLL_INTERVAL_SECONDS=$POLL_INTERVAL_SECONDS"
echo "MAX_GUARD_SECONDS=$MAX_GUARD_SECONDS"

deadline=$((SECONDS + MAX_GUARD_SECONDS))
while (( SECONDS < deadline )); do
  set +e
  bash scripts/hpc/guard_cancel_turin_after_b200_ready.sh
  status=$?
  set -e

  if [[ "$status" -eq "$NOT_READY_EXIT_CODE" ]]; then
    echo "Guard check not ready; sleeping ${POLL_INTERVAL_SECONDS}s."
    sleep "$POLL_INTERVAL_SECONDS"
    continue
  fi

  echo "Guard check exited with status $status."
  exit 0
done

echo "Guard max time reached without healthy B200 full-training metrics. Turin was left untouched."
