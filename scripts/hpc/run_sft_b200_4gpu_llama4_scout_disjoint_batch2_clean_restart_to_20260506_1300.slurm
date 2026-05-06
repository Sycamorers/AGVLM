#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-disjoint-clean-rs
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=512G
#SBATCH --time=15:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zijing.huang@ufl.edu

set -euo pipefail

cd /blue/hmedeiros/qinruoyao/agvlm

module load conda
module load cuda/12.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

disable_core_dumps() {
  ulimit -S -c 0 || true
  ulimit -H -c 0 || true
}
disable_core_dumps

export AGRI_VLM_DATA_ROOT="$PWD/data"
export HF_HOME="$PWD/.cache/huggingface"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$PWD/.tmp"
export TRITON_CACHE_DIR="$TMPDIR/triton-disjoint-clean-restart-$SLURM_JOB_ID"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export AGRI_VLM_REQUIRED_MODEL_ACCESS="meta-llama/Llama-4-Scout-17B-16E-Instruct"
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE=1048576

UPSTREAM_CHECKPOINT_DIR="/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft"
UPSTREAM_FALLBACK_CHECKPOINT_DIR="$UPSTREAM_CHECKPOINT_DIR/checkpoint-2250"
RUN_DIR="$PWD/outputs/sft/llama4-scout-17b-16e-lora-disjoint-b200-4gpu-batch2-clean-restart-to-20260506-1300"
GENERATED_CONFIG_DIR="$TMPDIR/generated-sft-clean-restart-configs-$SLURM_JOB_ID"
TRAIN_DEADLINE="${TRAIN_DEADLINE:-2026-05-06 13:00:00}"
TRAIN_DEADLINE_TZ="${TRAIN_DEADLINE_TZ:-America/New_York}"

mkdir -p \
  "$AGRI_VLM_DATA_ROOT" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$HF_XET_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TMPDIR" \
  "$TRITON_CACHE_DIR" \
  "$RUN_DIR" \
  "$RUN_DIR/data_validation" \
  "$GENERATED_CONFIG_DIR" \
  logs/slurm

python scripts/verify_environment.py

python scripts/data/validate_manifest_images.py \
  --manifest data/manifests/full/sft_train_llama4_disjoint_after_balanced_b200_agbase_max3.decode_valid.jsonl \
  --repo-root "$PWD" \
  --summary-output "$RUN_DIR/data_validation/train_decode_validation_summary.json" \
  --mode decode \
  --workers 16 \
  --progress-every 10000

python scripts/data/validate_manifest_images.py \
  --manifest data/manifests/full/sft_eval_llama4_max1_stratified512.decode_valid.jsonl \
  --repo-root "$PWD" \
  --summary-output "$RUN_DIR/data_validation/eval_decode_validation_summary.json" \
  --mode decode \
  --workers 8 \
  --progress-every 1000

UPSTREAM_SOURCE_DIR=""
if [[ -f "$UPSTREAM_CHECKPOINT_DIR/adapter_config.json" && -f "$UPSTREAM_CHECKPOINT_DIR/adapter_model.safetensors" ]]; then
  UPSTREAM_SOURCE_DIR="$UPSTREAM_CHECKPOINT_DIR"
elif [[ -f "$UPSTREAM_FALLBACK_CHECKPOINT_DIR/adapter_config.json" && -f "$UPSTREAM_FALLBACK_CHECKPOINT_DIR/adapter_model.safetensors" ]]; then
  UPSTREAM_SOURCE_DIR="$UPSTREAM_FALLBACK_CHECKPOINT_DIR"
else
  echo "Missing usable upstream adapter in $UPSTREAM_CHECKPOINT_DIR or $UPSTREAM_FALLBACK_CHECKPOINT_DIR"
  exit 2
fi
echo "Using upstream SFT adapter: $UPSTREAM_SOURCE_DIR"

TRAIN_DEADLINE_EPOCH="$(TZ="$TRAIN_DEADLINE_TZ" date -d "$TRAIN_DEADLINE" +%s)"
NOW_EPOCH="$(date +%s)"
TRAIN_TIMEOUT_SECONDS=$((TRAIN_DEADLINE_EPOCH - NOW_EPOCH))
if (( TRAIN_TIMEOUT_SECONDS <= 0 )); then
  echo "Training deadline has already passed: $TRAIN_DEADLINE $TRAIN_DEADLINE_TZ"
  exit 3
fi
echo "Training wall-clock deadline: $TRAIN_DEADLINE $TRAIN_DEADLINE_TZ (${TRAIN_TIMEOUT_SECONDS}s from now)"

TRAIN_CONFIG="$GENERATED_CONFIG_DIR/sft_lora_b200_4gpu_llama4_scout_disjoint_batch2_clean_restart_to_20260506_1300.yaml"
sed "s#^sft_checkpoint_path: .*#sft_checkpoint_path: $UPSTREAM_SOURCE_DIR#" \
  configs/train/sft_lora_b200_4gpu_llama4_scout_disjoint_batch2_clean_restart_to_20260506_1300.yaml > "$TRAIN_CONFIG"

TRAIN_EXIT=0
set +e
timeout --signal=TERM --kill-after=180s "${TRAIN_TIMEOUT_SECONDS}s" \
  python scripts/launch_torchrun.py \
  --nproc-per-node 4 \
  scripts/train/train_sft.py -- \
  --model-config configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml \
  --train-config "$TRAIN_CONFIG"
TRAIN_EXIT=$?
set -e
if [[ "$TRAIN_EXIT" -eq 124 || "$TRAIN_EXIT" -eq 137 ]]; then
  echo "Training stopped at the configured wall-clock deadline."
  TRAIN_EXIT=0
fi

if [[ -f "$RUN_DIR/metrics/train_metrics.jsonl" || -f "$RUN_DIR/metrics.jsonl" ]]; then
  python scripts/artifacts/export_training_artifacts.py --run-dir "$RUN_DIR" || true
fi

python scripts/artifacts/combine_sft_stage_results.py \
  --output outputs/sft/staged_sft_results/b200_disjoint_clean_restart_to_20260506_1300_results.json \
  --stage balanced_b200="$PWD/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft" \
  --stage disjoint_retry="$PWD/outputs/sft/llama4-scout-17b-16e-lora-disjoint-b200-4gpu-batch2-retry-to-20260506-1300" \
  --stage disjoint_clean_restart="$RUN_DIR" || true

rm -rf "$TRITON_CACHE_DIR"

echo "TensorBoard clean restart: tensorboard --logdir $RUN_DIR/tensorboard --host 0.0.0.0 --port 6012"
exit "$TRAIN_EXIT"
