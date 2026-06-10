#!/bin/bash -l
#SBATCH --job-name=agri-vlm-zero-adapter-recover
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${AGRI_VLM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

export PYTHONPATH=src
export TMPDIR="${AGRI_VLM_TMPDIR:-$PWD/.tmp/zero-adapter-recover-${SLURM_JOB_ID:-manual}}"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TMPDIR/xdg-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

NEW_SFT_ROOT="${NEW_SFT_ROOT:-/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1250,1500,1750,2000,2250,2500,2750}"
INCLUDE_FINAL_ROOT="${INCLUDE_FINAL_ROOT:-true}"
FINAL_STEP="${FINAL_STEP:-2813}"
ADAPTER_DTYPE="${ADAPTER_DTYPE:-bfloat16}"
RUN_TAG="${RUN_TAG:-recovered-new-sft-zero-adapters-${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference_checks/$RUN_TAG}"
REFERENCE_ADAPTER_CONFIG="${REFERENCE_ADAPTER_CONFIG:-$NEW_SFT_ROOT/adapter_config.json}"
REFERENCE_ADAPTER_MODEL="${REFERENCE_ADAPTER_MODEL:-$NEW_SFT_ROOT/adapter_model.safetensors}"
export NEW_SFT_ROOT CHECKPOINT_STEPS INCLUDE_FINAL_ROOT FINAL_STEP ADAPTER_DTYPE RUN_TAG OUTPUT_ROOT
export REFERENCE_ADAPTER_CONFIG REFERENCE_ADAPTER_MODEL

mkdir -p logs/slurm "$OUTPUT_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"

fail_fast() {
  echo "ERROR: $1" >&2
  exit 2
}

[[ -d "$NEW_SFT_ROOT" ]] || fail_fast "new SFT root missing: $NEW_SFT_ROOT"
[[ -f "$REFERENCE_ADAPTER_CONFIG" ]] || fail_fast "reference adapter config missing: $REFERENCE_ADAPTER_CONFIG"
[[ -f "$REFERENCE_ADAPTER_MODEL" ]] || fail_fast "reference adapter model missing: $REFERENCE_ADAPTER_MODEL"

echo "date=$(date -Is)"
echo "hostname=$(hostname -f)"
echo "NEW_SFT_ROOT=$NEW_SFT_ROOT"
echo "CHECKPOINT_STEPS=$CHECKPOINT_STEPS"
echo "INCLUDE_FINAL_ROOT=$INCLUDE_FINAL_ROOT"
echo "FINAL_STEP=$FINAL_STEP"
echo "ADAPTER_DTYPE=$ADAPTER_DTYPE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"

IFS=',' read -r -a steps <<< "$CHECKPOINT_STEPS"
for raw_step in "${steps[@]}"; do
  step="$(echo "$raw_step" | xargs)"
  [[ -n "$step" ]] || continue
  checkpoint_dir="$NEW_SFT_ROOT/checkpoint-$step"
  output_dir="$OUTPUT_ROOT/checkpoint-$step"
  [[ -d "$checkpoint_dir" ]] || fail_fast "checkpoint missing: $checkpoint_dir"
  echo "recovering_checkpoint=$checkpoint_dir"
  python scripts/eval/recover_peft_adapter_from_zero_checkpoint.py \
    --checkpoint-dir "$checkpoint_dir" \
    --output-dir "$output_dir" \
    --dtype "$ADAPTER_DTYPE" \
    --reference-adapter-config "$REFERENCE_ADAPTER_CONFIG" \
    --reference-adapter-model "$REFERENCE_ADAPTER_MODEL"
done

case "${INCLUDE_FINAL_ROOT,,}" in
  1|true|yes|on)
    final_output="$OUTPUT_ROOT/checkpoint-$FINAL_STEP"
    mkdir -p "$final_output"
    cp "$NEW_SFT_ROOT/adapter_config.json" "$final_output/adapter_config.json"
    [[ ! -f "$NEW_SFT_ROOT/README.md" ]] || cp "$NEW_SFT_ROOT/README.md" "$final_output/README.md"
    ln -sfn "$NEW_SFT_ROOT/adapter_model.safetensors" "$final_output/adapter_model.safetensors"
    ;;
  0|false|no|off)
    ;;
  *)
    fail_fast "INCLUDE_FINAL_ROOT must be boolean-like; got $INCLUDE_FINAL_ROOT"
    ;;
esac

python - <<'PY'
import json
import os
from pathlib import Path

from safetensors import safe_open

output_root = Path(os.environ["OUTPUT_ROOT"])
steps = [step.strip() for step in os.environ["CHECKPOINT_STEPS"].split(",") if step.strip()]
include_final = os.environ["INCLUDE_FINAL_ROOT"].strip().lower() in {"1", "true", "yes", "on"}
if include_final:
    final_step = os.environ["FINAL_STEP"].strip()
    if final_step not in steps:
        steps.append(final_step)

summary = []
for step in steps:
    checkpoint = output_root / f"checkpoint-{step}"
    adapter_model = checkpoint / "adapter_model.safetensors"
    adapter_config = checkpoint / "adapter_config.json"
    if not adapter_model.is_file():
        raise SystemExit(f"Missing recovered adapter_model.safetensors: {adapter_model}")
    if not adapter_config.is_file():
        raise SystemExit(f"Missing recovered adapter_config.json: {adapter_config}")
    with safe_open(adapter_model, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys:
            raise SystemExit(f"Recovered adapter has no tensors: {adapter_model}")
        tensor = handle.get_tensor(keys[0])
    summary.append(
        {
            "step": step,
            "checkpoint": str(checkpoint),
            "adapter_bytes": adapter_model.stat().st_size,
            "num_tensors": len(keys),
            "first_tensor": {
                "name": keys[0],
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            },
        }
    )

(output_root / "recovered_adapters.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"output_root": str(output_root), "adapters": summary}, indent=2, sort_keys=True))
PY

echo "adapter_recovery_completed=1"
echo "output_root=$OUTPUT_ROOT"
