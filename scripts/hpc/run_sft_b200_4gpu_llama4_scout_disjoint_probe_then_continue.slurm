#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-disjoint-b2
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=512G
#SBATCH --time=21:25:00
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
export TRITON_CACHE_DIR="$TMPDIR/triton-disjoint-$SLURM_JOB_ID"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export AGRI_VLM_REQUIRED_MODEL_ACCESS="meta-llama/Llama-4-Scout-17B-16E-Instruct"
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

UPSTREAM_CHECKPOINT_DIR="/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft"
UPSTREAM_FALLBACK_CHECKPOINT_DIR="$UPSTREAM_CHECKPOINT_DIR/checkpoint-2250"
PROBE_RUN_DIR="$PWD/outputs/sft/llama4-scout-17b-16e-lora-disjoint-b200-4gpu-batch2-probe"
BATCH2_RUN_DIR="$PWD/outputs/sft/llama4-scout-17b-16e-lora-disjoint-b200-4gpu-batch2-continuation"
BATCH1_RUN_DIR="$PWD/outputs/sft/llama4-scout-17b-16e-lora-disjoint-b200-4gpu-batch1-fallback"
GENERATED_CONFIG_DIR="$TMPDIR/generated-sft-configs-$SLURM_JOB_ID"

mkdir -p \
  "$AGRI_VLM_DATA_ROOT" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$HF_XET_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TMPDIR" \
  "$TRITON_CACHE_DIR" \
  "$PROBE_RUN_DIR" \
  "$BATCH2_RUN_DIR" \
  "$BATCH1_RUN_DIR" \
  "$GENERATED_CONFIG_DIR" \
  logs/slurm

python scripts/verify_environment.py
python scripts/data/build_disjoint_sft_manifest.py \
  --config configs/data/sft_train_llama4_disjoint_after_balanced_b200_agbase_max3.yaml

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

prepare_train_config() {
  local source_config="$1"
  local output_config="$2"
  sed "s#^sft_checkpoint_path: .*#sft_checkpoint_path: $UPSTREAM_SOURCE_DIR#" \
    "$source_config" > "$output_config"
}

PROBE_CONFIG="$GENERATED_CONFIG_DIR/sft_lora_b200_4gpu_llama4_scout_disjoint_batch2_probe.yaml"
BATCH2_CONFIG="$GENERATED_CONFIG_DIR/sft_lora_b200_4gpu_llama4_scout_disjoint_batch2_continuation.yaml"
BATCH1_CONFIG="$GENERATED_CONFIG_DIR/sft_lora_b200_4gpu_llama4_scout_disjoint_batch1_fallback.yaml"
prepare_train_config configs/train/sft_lora_b200_4gpu_llama4_scout_disjoint_batch2_probe.yaml "$PROBE_CONFIG"
prepare_train_config configs/train/sft_lora_b200_4gpu_llama4_scout_disjoint_batch2_continuation.yaml "$BATCH2_CONFIG"
prepare_train_config configs/train/sft_lora_b200_4gpu_llama4_scout_disjoint_batch1_fallback.yaml "$BATCH1_CONFIG"

PROBE_EXIT=0
set +e
python scripts/launch_torchrun.py \
  --nproc-per-node 4 \
  scripts/train/train_sft.py -- \
  --model-config configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml \
  --train-config "$PROBE_CONFIG"
PROBE_EXIT=$?
set -e

if [[ "$PROBE_EXIT" -eq 0 ]]; then
  TRAIN_CONFIG="$BATCH2_CONFIG"
  FINAL_RUN_DIR="$BATCH2_RUN_DIR"
  echo "Batch-2 probe passed; continuing with $TRAIN_CONFIG"
else
  TRAIN_CONFIG="$BATCH1_CONFIG"
  FINAL_RUN_DIR="$BATCH1_RUN_DIR"
  echo "Batch-2 probe failed with exit $PROBE_EXIT; falling back to $TRAIN_CONFIG"
fi

TRAIN_EXIT=0
set +e
python scripts/launch_torchrun.py \
  --nproc-per-node 4 \
  scripts/train/train_sft.py -- \
  --model-config configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml \
  --train-config "$TRAIN_CONFIG"
TRAIN_EXIT=$?
set -e

for RUN_DIR in "$PROBE_RUN_DIR" "$FINAL_RUN_DIR"; do
  if [[ -f "$RUN_DIR/metrics/train_metrics.jsonl" || -f "$RUN_DIR/metrics.jsonl" ]]; then
    python scripts/artifacts/export_training_artifacts.py --run-dir "$RUN_DIR" || true
  fi
done

python scripts/artifacts/combine_sft_stage_results.py \
  --output outputs/sft/staged_sft_results/b200_disjoint_continuation_results.json \
  --stage balanced_b200="$PWD/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft" \
  --stage disjoint_probe="$PROBE_RUN_DIR" \
  --stage disjoint_continuation="$FINAL_RUN_DIR" || true

rm -rf "$TRITON_CACHE_DIR"

echo "TensorBoard probe: tensorboard --logdir $PROBE_RUN_DIR/tensorboard --host 0.0.0.0 --port 6009"
echo "TensorBoard continuation: tensorboard --logdir $FINAL_RUN_DIR/tensorboard --host 0.0.0.0 --port 6010"
exit "$TRAIN_EXIT"
