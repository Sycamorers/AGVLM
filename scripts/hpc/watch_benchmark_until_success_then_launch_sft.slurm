#!/bin/bash -l
#SBATCH --job-name=agri-watch-bench-sft
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd /blue/hmedeiros/qinruoyao/agvlm

MODEL_KEY="${MODEL_KEY:-agvlm_phi4_sft_completed}"
QUANTIZATION="${QUANTIZATION:-4bit}"
BENCHMARK_SLURM="${BENCHMARK_SLURM:-benchmarks/vlm_baselines/slurm/run_sft_benchmark_agvlm_checkpoint.sbatch}"
BENCHMARK_OUTPUT_BASE="${BENCHMARK_OUTPUT_BASE:-benchmarks/vlm_baselines/results/agvlm_previous_sft_benchmark_watch}"
ATTEMPT_SLEEP_SECONDS="${ATTEMPT_SLEEP_SECONDS:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
RETRY_ON_GATE_REJECT="${RETRY_ON_GATE_REJECT:-runtime_failures}"

TRAIN_SLURM="${TRAIN_SLURM:-scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/sft_train_eval_phi4_max3.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/phi4_reasoning_vision_15b_turin_24g.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_balanced_v2_instructional_full.yaml}"
PREFLIGHT_CONFIG="${PREFLIGHT_CONFIG:-configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_balanced_v2_instructional_preflight.yaml}"

REQUIRE_NUM_EXAMPLES_MIN="${REQUIRE_NUM_EXAMPLES_MIN:-392}"
REQUIRE_FAILURE_RATE_MAX="${REQUIRE_FAILURE_RATE_MAX:-0.0}"
REQUIRE_INVALID_RATE_MAX="${REQUIRE_INVALID_RATE_MAX:-0.50}"
REQUIRE_TASK_MACRO_MIN="${REQUIRE_TASK_MACRO_MIN:-0.13}"
REQUIRE_VQA_RELAXED_MIN="${REQUIRE_VQA_RELAXED_MIN:-0.12}"
REQUIRE_CLARIFY_MACRO_F1_MIN="${REQUIRE_CLARIFY_MACRO_F1_MIN:-0.20}"

WATCH_REPORT_DIR="${WATCH_REPORT_DIR:-reports/sft_regression_audit}"
mkdir -p logs/slurm "$WATCH_REPORT_DIR" "$BENCHMARK_OUTPUT_BASE"

attempt=0
while true; do
  attempt=$((attempt + 1))
  if [[ "$MAX_ATTEMPTS" != "0" && "$attempt" -gt "$MAX_ATTEMPTS" ]]; then
    echo "max_attempts_reached=$MAX_ATTEMPTS"
    exit 2
  fi

  output_dir="${BENCHMARK_OUTPUT_BASE}/attempt-${attempt}"
  mkdir -p "$output_dir"
  echo "benchmark_attempt=$attempt output_dir=$output_dir"
  benchmark_job="$(sbatch --parsable --export=ALL,MODEL_KEY="$MODEL_KEY",OUTPUT_DIR="$output_dir",QUANTIZATION="$QUANTIZATION" "$BENCHMARK_SLURM")"
  echo "benchmark_job=$benchmark_job"

  while true; do
    state_line="$(sacct -j "$benchmark_job" -X --format=State,ExitCode,Elapsed -P -n 2>/dev/null | head -n 1 || true)"
    state="$(cut -d'|' -f1 <<<"$state_line")"
    exit_code="$(cut -d'|' -f2 <<<"$state_line")"
    if [[ "$state" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)$ ]]; then
      break
    fi
    echo "benchmark_job=$benchmark_job state=${state:-pending} elapsed=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sleep "$ATTEMPT_SLEEP_SECONDS"
  done

  echo "benchmark_job=$benchmark_job final_state=$state exit_code=$exit_code"
  if [[ "$state" == "COMPLETED" && "$exit_code" == "0:0" ]]; then
    gate_json="$WATCH_REPORT_DIR/benchmark_gate_then_sft_${benchmark_job}.json"
    gate_md="$WATCH_REPORT_DIR/benchmark_gate_then_sft_${benchmark_job}.md"
    python3 scripts/hpc/benchmark_gate_then_submit_sft.py \
      --benchmark-job-id "$benchmark_job" \
      --benchmark-output-dir "$output_dir" \
      --model-key "$MODEL_KEY" \
      --min-num-examples "$REQUIRE_NUM_EXAMPLES_MIN" \
      --max-failure-rate "$REQUIRE_FAILURE_RATE_MAX" \
      --max-invalid-rate "$REQUIRE_INVALID_RATE_MAX" \
      --min-task-macro "$REQUIRE_TASK_MACRO_MIN" \
      --min-vqa-relaxed "$REQUIRE_VQA_RELAXED_MIN" \
      --min-clarify-macro-f1 "$REQUIRE_CLARIFY_MACRO_F1_MIN" \
      --train-slurm "$TRAIN_SLURM" \
      --data-config "$DATA_CONFIG" \
      --model-config "$MODEL_CONFIG" \
      --train-config "$TRAIN_CONFIG" \
      --preflight-config "$PREFLIGHT_CONFIG" \
      --output-json "$gate_json" \
      --output-md "$gate_md"
    decision="$(python3 - "$gate_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
print(payload.get("decision", ""))
PY
)"
    echo "gate_report=$gate_md decision=$decision"
    if [[ "$decision" == "submit_training" ]]; then
      echo "benchmark_success_and_training_submitted=1"
      exit 0
    fi
    if [[ "$RETRY_ON_GATE_REJECT" != "none" ]]; then
      failure_rate="$(python3 - "$gate_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
value = payload.get("metrics", {}).get("failure_rate")
print("" if value is None else value)
PY
)"
      retry_gate_reject=0
      if [[ "$RETRY_ON_GATE_REJECT" == "always" ]]; then
        retry_gate_reject=1
      elif [[ -n "$failure_rate" ]]; then
        retry_gate_reject="$(python3 - "$failure_rate" "$REQUIRE_FAILURE_RATE_MAX" <<'PY'
import sys
actual = float(sys.argv[1])
required = float(sys.argv[2])
print(1 if actual > required else 0)
PY
)"
      fi
      if [[ "$retry_gate_reject" == "1" ]]; then
        echo "benchmark_completed_but_runtime_failures_exceeded_gate=1 failure_rate=${failure_rate}"
        echo "retrying_after_gate_reject=1"
        sleep "$ATTEMPT_SLEEP_SECONDS"
        continue
      fi
    fi
    echo "benchmark_completed_but_gate_rejected=1"
    exit 3
  fi

  err_path="benchmarks/vlm_baselines/logs/slurm/agri-sft-ckpt-bench-${benchmark_job}.err"
  if [[ -f "$err_path" ]]; then
    echo "last_error_log=$err_path"
    tail -80 "$err_path" || true
  fi
  echo "retrying_after_failed_benchmark=1"
done
