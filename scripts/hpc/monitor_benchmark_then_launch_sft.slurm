#!/bin/bash -l
#SBATCH --job-name=agri-bench-gate-sft
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd /blue/hmedeiros/qinruoyao/agvlm

BENCHMARK_JOB_ID="${BENCHMARK_JOB_ID:?Set BENCHMARK_JOB_ID to the Slurm benchmark job id.}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:?Set BENCHMARK_OUTPUT_DIR to the benchmark result directory.}"
MODEL_KEY="${MODEL_KEY:-agvlm_phi4_sft_completed}"

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

MONITOR_OUTPUT_JSON="${MONITOR_OUTPUT_JSON:-reports/sft_regression_audit/benchmark_gate_then_sft_${BENCHMARK_JOB_ID}.json}"
MONITOR_OUTPUT_MD="${MONITOR_OUTPUT_MD:-reports/sft_regression_audit/benchmark_gate_then_sft_${BENCHMARK_JOB_ID}.md}"

mkdir -p logs/slurm "$(dirname "$MONITOR_OUTPUT_JSON")"

args=(
  scripts/hpc/benchmark_gate_then_submit_sft.py
  --benchmark-job-id "$BENCHMARK_JOB_ID"
  --benchmark-output-dir "$BENCHMARK_OUTPUT_DIR"
  --model-key "$MODEL_KEY"
  --min-num-examples "$REQUIRE_NUM_EXAMPLES_MIN"
  --max-failure-rate "$REQUIRE_FAILURE_RATE_MAX"
  --max-invalid-rate "$REQUIRE_INVALID_RATE_MAX"
  --min-task-macro "$REQUIRE_TASK_MACRO_MIN"
  --min-vqa-relaxed "$REQUIRE_VQA_RELAXED_MIN"
  --min-clarify-macro-f1 "$REQUIRE_CLARIFY_MACRO_F1_MIN"
  --train-slurm "$TRAIN_SLURM"
  --data-config "$DATA_CONFIG"
  --model-config "$MODEL_CONFIG"
  --train-config "$TRAIN_CONFIG"
  --preflight-config "$PREFLIGHT_CONFIG"
  --output-json "$MONITOR_OUTPUT_JSON"
  --output-md "$MONITOR_OUTPUT_MD"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

python3 "${args[@]}"
