#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/blue/hmedeiros/qinruoyao/agvlm"
B200_SCRIPT="scripts/hpc/run_sft_b200_4gpu_phi4_reasoning_vision_15b_full_max3.slurm"
GUARD_SCRIPT="scripts/hpc/guard_cancel_turin_after_b200_ready.slurm"
B200_RUN_DIR="outputs/sft/phi4-reasoning-vision-15b-full-max3-b200-4gpu"
B200_JOB_NAME="agri-vlm-sft-phi4rv-b200x4"

CLI_TURIN_JOB_ID=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --turin-job-id)
      CLI_TURIN_JOB_ID="${2:-}"
      shift 2
      ;;
    --turin-job-id=*)
      CLI_TURIN_JOB_ID="${1#*=}"
      shift
      ;;
    *)
      if [[ -z "$CLI_TURIN_JOB_ID" ]]; then
        CLI_TURIN_JOB_ID="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        exit 2
      fi
      ;;
  esac
done

cd "$REPO_ROOT"
if [[ "$PWD" != "$REPO_ROOT" ]]; then
  echo "ERROR: expected repo path $REPO_ROOT, got $PWD" >&2
  exit 2
fi

echo "Repository: $PWD"
echo "Git status:"
git status --short --branch

TURIN_JOB_ID="${TURIN_JOB_ID:-$CLI_TURIN_JOB_ID}"
if [[ -z "$TURIN_JOB_ID" ]]; then
  mapfile -t candidates < <(
    squeue -h -u "$USER" -o "%i|%P|%j|%T|%D|%C|%R" \
      | awk -F'|' '$2 == "hpg-turin" && $4 == "RUNNING" && $3 ~ /phi4rv|phi4|Phi-4/ {print}'
  )
  if [[ "${#candidates[@]}" -eq 1 ]]; then
    TURIN_JOB_ID="$(printf '%s\n' "${candidates[0]}" | awk -F'|' '{print $1}')"
  elif [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "ERROR: no running Turin Phi-4 SFT job found. Provide TURIN_JOB_ID." >&2
    exit 2
  else
    echo "ERROR: multiple running Turin Phi-4 candidates found. Provide TURIN_JOB_ID explicitly." >&2
    printf '%s\n' "${candidates[@]}" >&2
    exit 2
  fi
fi

echo "Selected TURIN_JOB_ID=$TURIN_JOB_ID"
TURIN_INFO="$(squeue -h -j "$TURIN_JOB_ID" -o "%i|%P|%j|%T|%D|%C|%R" | head -n 1)"
if [[ -z "$TURIN_INFO" ]]; then
  echo "ERROR: selected Turin job $TURIN_JOB_ID is not present in squeue." >&2
  exit 2
fi
TURIN_PARTITION="$(printf '%s\n' "$TURIN_INFO" | awk -F'|' '{print $2}')"
TURIN_NAME="$(printf '%s\n' "$TURIN_INFO" | awk -F'|' '{print $3}')"
TURIN_STATE="$(printf '%s\n' "$TURIN_INFO" | awk -F'|' '{print $4}')"
if [[ "$TURIN_PARTITION" != "hpg-turin" || "$TURIN_STATE" != "RUNNING" || ! "$TURIN_NAME" =~ (phi4rv|phi4|Phi-4) ]]; then
  echo "ERROR: selected job is not the expected running Turin Phi-4 job: $TURIN_INFO" >&2
  exit 2
fi
squeue -j "$TURIN_JOB_ID" -o "%.18i %.18P %.40j %.8T %.8D %.8C %.10m %.20R"

echo "Submitting B200 training job: sbatch $B200_SCRIPT"
set +e
B200_SUBMIT_OUTPUT="$(sbatch "$B200_SCRIPT" 2>&1)"
B200_SUBMIT_STATUS=$?
set -e
echo "$B200_SUBMIT_OUTPUT"
if [[ "$B200_SUBMIT_STATUS" -ne 0 ]]; then
  echo "ERROR: B200 submission failed. Turin job $TURIN_JOB_ID was left untouched." >&2
  exit "$B200_SUBMIT_STATUS"
fi

B200_JOB_ID="$(printf '%s\n' "$B200_SUBMIT_OUTPUT" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
if [[ -z "$B200_JOB_ID" ]]; then
  echo "ERROR: could not parse B200 job ID. Turin job $TURIN_JOB_ID was left untouched." >&2
  exit 2
fi
echo "B200_JOB_ID=$B200_JOB_ID"

GUARD_EXPORT="ALL,B200_JOB_ID=${B200_JOB_ID},TURIN_JOB_ID=${TURIN_JOB_ID},B200_RUN_DIR=${B200_RUN_DIR}"
echo "Submitting guard job: sbatch --dependency=after:${B200_JOB_ID} --export=${GUARD_EXPORT} $GUARD_SCRIPT"
set +e
GUARD_SUBMIT_OUTPUT="$(sbatch --dependency=after:"$B200_JOB_ID" --export="$GUARD_EXPORT" "$GUARD_SCRIPT" 2>&1)"
GUARD_SUBMIT_STATUS=$?
set -e
echo "$GUARD_SUBMIT_OUTPUT"
if [[ "$GUARD_SUBMIT_STATUS" -ne 0 ]]; then
  echo "ERROR: guard submission failed. Turin job $TURIN_JOB_ID was left untouched." >&2
  exit "$GUARD_SUBMIT_STATUS"
fi

GUARD_JOB_ID="$(printf '%s\n' "$GUARD_SUBMIT_OUTPUT" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
if [[ -z "$GUARD_JOB_ID" ]]; then
  echo "ERROR: could not parse guard job ID. Turin job $TURIN_JOB_ID was left untouched." >&2
  exit 2
fi
echo "GUARD_JOB_ID=$GUARD_JOB_ID"

echo "Monitor commands:"
echo "squeue -j \"$B200_JOB_ID\""
echo "squeue -j \"$TURIN_JOB_ID\""
echo "squeue -j \"$GUARD_JOB_ID\""
echo "tail -f logs/slurm/${B200_JOB_NAME}-${B200_JOB_ID}.out"
echo "tail -f logs/slurm/${B200_JOB_NAME}-${B200_JOB_ID}.err"
