#!/usr/bin/env bash
set -euo pipefail

B200_JOB_ID="${B200_JOB_ID:-${1:-}}"
TURIN_JOB_ID="${TURIN_JOB_ID:-${2:-}}"
B200_RUN_DIR="${B200_RUN_DIR:-${3:-}}"
MIN_METRIC_ROWS="${MIN_METRIC_ROWS:-${4:-3}}"
MAX_METRIC_STALE_SECONDS="${MAX_METRIC_STALE_SECONDS:-0}"
NOT_READY_EXIT_CODE="${NOT_READY_EXIT_CODE:-10}"

no_cancel() {
  echo "GUARD_RESULT=no_cancel"
  echo "GUARD_REASON=$1"
  exit 0
}

not_ready() {
  echo "GUARD_RESULT=not_ready"
  echo "GUARD_REASON=$1"
  exit "$NOT_READY_EXIT_CODE"
}

squeue_state() {
  local job_id="$1"
  squeue -h -j "$job_id" -o "%T" 2>/dev/null | head -n 1 | tr -d ' ' || true
}

sacct_state() {
  local job_id="$1"
  sacct -n -X -j "$job_id" -o State 2>/dev/null | head -n 1 | awk '{print $1}' || true
}

job_start_epoch() {
  local job_id="$1"
  local start_time
  start_time="$(squeue -h -j "$job_id" -o "%S" 2>/dev/null | head -n 1 | awk '{print $1}')"
  if [[ -z "$start_time" || "$start_time" == "N/A" || "$start_time" == "Unknown" ]]; then
    return 0
  fi
  date -d "$start_time" +%s 2>/dev/null || true
}

if [[ -z "$B200_JOB_ID" || -z "$TURIN_JOB_ID" || -z "$B200_RUN_DIR" ]]; then
  no_cancel "missing required input; need B200_JOB_ID, TURIN_JOB_ID, and B200_RUN_DIR"
fi

if [[ "$B200_RUN_DIR" == *preflight* ]]; then
  no_cancel "B200_RUN_DIR appears to be a preflight path: $B200_RUN_DIR"
fi

case "$MIN_METRIC_ROWS" in
  ''|*[!0-9]*) no_cancel "MIN_METRIC_ROWS must be an integer: $MIN_METRIC_ROWS" ;;
esac

TURIN_STATE="$(squeue_state "$TURIN_JOB_ID")"
if [[ -z "$TURIN_STATE" ]]; then
  no_cancel "Turin job $TURIN_JOB_ID is not present in squeue"
fi
if [[ "$TURIN_STATE" != "RUNNING" ]]; then
  no_cancel "Turin job $TURIN_JOB_ID is not RUNNING; current state is $TURIN_STATE"
fi

B200_STATE="$(squeue_state "$B200_JOB_ID")"
if [[ -z "$B200_STATE" ]]; then
  B200_SACCT_STATE="$(sacct_state "$B200_JOB_ID")"
  no_cancel "B200 job $B200_JOB_ID is not present in squeue; sacct state is ${B200_SACCT_STATE:-unknown}"
fi

case "$B200_STATE" in
  RUNNING)
    ;;
  PENDING|CONFIGURING)
    not_ready "B200 job $B200_JOB_ID is not RUNNING yet; current state is $B200_STATE"
    ;;
  FAILED|CANCELLED|CANCELLED+|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
    no_cancel "B200 job $B200_JOB_ID is in terminal/suspicious state $B200_STATE"
    ;;
  *)
    no_cancel "B200 job $B200_JOB_ID is not RUNNING; current state is $B200_STATE"
    ;;
esac

B200_START_EPOCH="$(job_start_epoch "$B200_JOB_ID")"

set +e
METRIC_SUMMARY="$(
  python - "$B200_RUN_DIR" "$MIN_METRIC_ROWS" "${B200_START_EPOCH:-}" "$MAX_METRIC_STALE_SECONDS" <<'PY'
import json
import os
from pathlib import Path
import sys
import time

run_dir = Path(sys.argv[1])
min_rows = int(sys.argv[2])
start_epoch = int(sys.argv[3]) if sys.argv[3].strip() else None
max_stale_seconds = int(sys.argv[4]) if sys.argv[4].strip() else 0

if "preflight" in {part.lower() for part in run_dir.parts}:
    print("metrics_error=run_dir_is_preflight")
    raise SystemExit(10)

if run_dir.is_absolute():
    resolved_run_dir = run_dir.resolve()
else:
    resolved_run_dir = (Path.cwd() / run_dir).resolve()

try:
    relative_parts = resolved_run_dir.relative_to(Path.cwd().resolve()).parts
except ValueError:
    relative_parts = resolved_run_dir.parts

if "outputs" not in relative_parts or "sft" not in relative_parts:
    print("metrics_error=run_dir_not_under_outputs_sft")
    raise SystemExit(10)
if not any("b200" in part.lower() for part in relative_parts):
    print("metrics_error=run_dir_not_b200")
    raise SystemExit(10)

candidates = [
    resolved_run_dir / "metrics" / "train_metrics.jsonl",
    resolved_run_dir / "metrics.jsonl",
]
progress_keys = {
    "step",
    "global_step",
    "loss",
    "train_loss",
    "learning_rate",
    "epoch",
    "grad_norm",
}

errors = []
for candidate in candidates:
    if not candidate.exists():
        errors.append(f"{candidate}:missing")
        continue
    if candidate.stat().st_size == 0:
        errors.append(f"{candidate}:empty")
        continue
    mtime = candidate.stat().st_mtime
    if start_epoch is not None and mtime + 1 < start_epoch:
        errors.append(f"{candidate}:mtime_before_b200_start")
        continue
    if max_stale_seconds > 0 and time.time() - mtime > max_stale_seconds:
        errors.append(f"{candidate}:stale")
        continue

    rows = []
    with candidate.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"{candidate}:invalid_json_line_{line_number}:{exc.msg}")
                rows = []
                break
            if not isinstance(payload, dict):
                errors.append(f"{candidate}:non_object_line_{line_number}")
                rows = []
                break
            rows.append(payload)
    if not rows:
        continue

    progress_rows = [row for row in rows if progress_keys.intersection(row)]
    if len(rows) < min_rows:
        errors.append(f"{candidate}:rows_{len(rows)}_lt_{min_rows}")
        continue
    if len(progress_rows) < min_rows:
        errors.append(f"{candidate}:progress_rows_{len(progress_rows)}_lt_{min_rows}")
        continue

    last = progress_rows[-1]
    summary_keys = [
        "global_step",
        "step",
        "loss",
        "train_loss",
        "learning_rate",
        "epoch",
        "grad_norm",
    ]
    last_summary = {key: last[key] for key in summary_keys if key in last}
    print("metrics_ready=1")
    print(f"metrics_path={candidate}")
    print(f"valid_metric_rows={len(rows)}")
    print(f"progress_metric_rows={len(progress_rows)}")
    print(f"metrics_mtime_epoch={int(mtime)}")
    if start_epoch is not None:
        print(f"b200_start_epoch={start_epoch}")
    print("last_metric_row_summary=" + json.dumps(last_summary, sort_keys=True))
    raise SystemExit(0)

print("metrics_ready=0")
print("metrics_errors=" + ";".join(errors))
raise SystemExit(10)
PY
)"
METRIC_STATUS=$?
set -e

echo "$METRIC_SUMMARY"
if [[ "$METRIC_STATUS" -ne 0 ]]; then
  not_ready "B200 full-training metrics are not ready"
fi

echo "B200_READINESS_SUMMARY_BEGIN"
echo "b200_job_id=$B200_JOB_ID"
echo "b200_state=$B200_STATE"
echo "turin_job_id=$TURIN_JOB_ID"
echo "turin_state_before=$TURIN_STATE"
echo "b200_run_dir=$B200_RUN_DIR"
echo "$METRIC_SUMMARY"
echo "B200_READINESS_SUMMARY_END"

echo "Running: scancel $TURIN_JOB_ID"
scancel "$TURIN_JOB_ID"
sleep 5

TURIN_STATE_AFTER="$(squeue_state "$TURIN_JOB_ID")"
if [[ -n "$TURIN_STATE_AFTER" ]]; then
  echo "turin_state_after_scancel=$TURIN_STATE_AFTER"
else
  TURIN_SACCT_AFTER="$(sacct_state "$TURIN_JOB_ID")"
  echo "turin_state_after_scancel=${TURIN_SACCT_AFTER:-not_in_squeue}"
fi
echo "GUARD_RESULT=cancelled_turin"
