#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-ckpt-sweep
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd /blue/hmedeiros/qinruoyao/agvlm

module load conda
module load cuda/12.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

export PYTHONPATH=src
export AGRI_VLM_DATA_ROOT="${AGRI_VLM_DATA_ROOT:-$PWD/data}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/transformers}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export TMPDIR="${AGRI_VLM_TMPDIR:-$PWD/.tmp/sft-checkpoint-sweep-${SLURM_JOB_ID:-manual}}"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TMPDIR/xdg-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/phi4_reasoning_vision_15b_turin_24g_4bit_rl.yaml}"
PREVIOUS_SFT_CHECKPOINT_PATH="${PREVIOUS_SFT_CHECKPOINT_PATH:-/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1}"
NEW_SFT_ROOT="${NEW_SFT_ROOT:-/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1250,1500,1750,2000,2250,2500,2750,2813}"
EVAL_MANIFEST_PATH="${EVAL_MANIFEST_PATH:-data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-128}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
EVAL_MIN_NEW_TOKENS="${EVAL_MIN_NEW_TOKENS:-2}"
EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-false}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_SHARDS="${NUM_SHARDS:-3}"
RUN_TAG="${RUN_TAG:-sft-checkpoint-sweep-new-round-${EVAL_MAX_EXAMPLES}-turin-${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference_checks/$RUN_TAG}"
SELECTED_MANIFEST_PATH="$OUTPUT_ROOT/manifests/selected_eval_manifest.jsonl"
MODEL_SPECS_PATH="$OUTPUT_ROOT/model_specs.json"
COMPARISON_PATH="$OUTPUT_ROOT/comparison.jsonl"
METRICS_PATH="$OUTPUT_ROOT/multi_model_metrics.json"
SUMMARY_PATH="$OUTPUT_ROOT/summary.md"
PAIRWISE_REPORT_PATH="$OUTPUT_ROOT/pairwise_comparison.md"
GENERATION_LABEL="${GENERATION_LABEL:-greedy, min_new_tokens=${EVAL_MIN_NEW_TOKENS}, max_new_tokens=${EVAL_MAX_NEW_TOKENS}, 4-bit Turin/L4 checkpoint sweep}"
export MODEL_CONFIG PREVIOUS_SFT_CHECKPOINT_PATH NEW_SFT_ROOT CHECKPOINT_STEPS
export EVAL_MANIFEST_PATH EVAL_MAX_EXAMPLES EVAL_MAX_NEW_TOKENS EVAL_MIN_NEW_TOKENS
export EVAL_DO_SAMPLE EVAL_BATCH_SIZE NUM_SHARDS RUN_TAG OUTPUT_ROOT SELECTED_MANIFEST_PATH
export MODEL_SPECS_PATH COMPARISON_PATH METRICS_PATH SUMMARY_PATH PAIRWISE_REPORT_PATH GENERATION_LABEL

mkdir -p \
  logs/slurm \
  "$AGRI_VLM_DATA_ROOT" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$HF_XET_CACHE" \
  "$TMPDIR" \
  "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$OUTPUT_ROOT/logs"

fail_fast() {
  echo "ERROR: $1" >&2
  exit 2
}

[[ -f "$MODEL_CONFIG" ]] || fail_fast "model config missing: $MODEL_CONFIG"
[[ -f "$EVAL_MANIFEST_PATH" ]] || fail_fast "eval manifest missing: $EVAL_MANIFEST_PATH"
[[ -d "$PREVIOUS_SFT_CHECKPOINT_PATH" ]] || fail_fast "previous SFT checkpoint path missing: $PREVIOUS_SFT_CHECKPOINT_PATH"
[[ -d "$NEW_SFT_ROOT" ]] || fail_fast "new SFT root missing: $NEW_SFT_ROOT"
[[ "$NUM_SHARDS" -le 3 ]] || fail_fast "NUM_SHARDS must be <=3 for this Turin wrapper; got $NUM_SHARDS"
[[ "$NUM_SHARDS" -gt 0 ]] || fail_fast "NUM_SHARDS must be positive"

python - <<'PY'
import json
import os
from pathlib import Path

from safetensors import safe_open

previous = Path(os.environ["PREVIOUS_SFT_CHECKPOINT_PATH"])
new_root = Path(os.environ["NEW_SFT_ROOT"])
steps = [step.strip() for step in os.environ["CHECKPOINT_STEPS"].split(",") if step.strip()]
specs = [
    {
        "key": "previous_sft",
        "display_name": "Previous SFT",
        "checkpoint_type": "sft",
        "checkpoint_path": str(previous),
    }
]
for step in steps:
    checkpoint = new_root / f"checkpoint-{step}"
    specs.append(
        {
            "key": f"new_sft_ckpt_{step}",
            "display_name": f"New SFT ckpt-{step}",
            "checkpoint_type": "sft",
            "checkpoint_path": str(checkpoint),
        }
    )

for spec in specs:
    checkpoint = Path(spec["checkpoint_path"])
    adapter_config_path = checkpoint / "adapter_config.json"
    adapter_model_path = checkpoint / "adapter_model.safetensors"
    if not adapter_config_path.is_file():
        raise SystemExit(f"{spec['key']}: adapter_config.json missing under {checkpoint}")
    if not adapter_model_path.is_file():
        raise SystemExit(f"{spec['key']}: adapter_model.safetensors missing under {checkpoint}")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    with safe_open(adapter_model_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys:
            raise SystemExit(f"{spec['key']}: adapter_model.safetensors has no tensors: {adapter_model_path}")
        first_tensor = handle.get_tensor(keys[0])
    print(
        json.dumps(
            {
                "key": spec["key"],
                "checkpoint": str(checkpoint),
                "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
                "peft_type": adapter_config.get("peft_type"),
                "r": adapter_config.get("r"),
                "lora_alpha": adapter_config.get("lora_alpha"),
                "num_tensors": len(keys),
                "first_tensor": {
                    "name": keys[0],
                    "shape": list(first_tensor.shape),
                    "dtype": str(first_tensor.dtype),
                },
                "adapter_bytes": adapter_model_path.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        )
    )

Path(os.environ["MODEL_SPECS_PATH"]).parent.mkdir(parents=True, exist_ok=True)
Path(os.environ["MODEL_SPECS_PATH"]).write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python - <<'PY'
import json
import os
from pathlib import Path

import yaml

eval_manifest_path = Path(os.environ["EVAL_MANIFEST_PATH"])
output_root = Path(os.environ["OUTPUT_ROOT"])
selected_manifest_path = Path(os.environ["SELECTED_MANIFEST_PATH"])
model_specs = json.loads(Path(os.environ["MODEL_SPECS_PATH"]).read_text(encoding="utf-8"))
num_shards = int(os.environ["NUM_SHARDS"])
max_examples = int(os.environ["EVAL_MAX_EXAMPLES"])
batch_size = int(os.environ["EVAL_BATCH_SIZE"])
max_new_tokens = int(os.environ["EVAL_MAX_NEW_TOKENS"])
min_new_tokens = int(os.environ["EVAL_MIN_NEW_TOKENS"])
do_sample_text = os.environ["EVAL_DO_SAMPLE"].strip().lower()
if do_sample_text in {"1", "true", "yes", "on"}:
    do_sample = True
elif do_sample_text in {"0", "false", "no", "off"}:
    do_sample = False
else:
    raise SystemExit(f"EVAL_DO_SAMPLE must be boolean-like; got {os.environ['EVAL_DO_SAMPLE']!r}")

rows = [
    json.loads(line)
    for line in eval_manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if max_examples > 0:
    rows = rows[:max_examples]
if not rows:
    raise SystemExit(f"No evaluation rows selected from {eval_manifest_path}")

sample_ids = [str(row.get("sample_id") or "") for row in rows]
duplicates = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
if duplicates:
    raise SystemExit(f"Duplicate sample ids in selected eval rows: {duplicates[:10]}")

manifest_dir = output_root / "manifests"
config_root = output_root / "configs"
manifest_dir.mkdir(parents=True, exist_ok=True)
config_root.mkdir(parents=True, exist_ok=True)
selected_manifest_path.parent.mkdir(parents=True, exist_ok=True)
selected_manifest_path.write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    encoding="utf-8",
)

for shard_id in range(num_shards):
    shard_rows = [row for index, row in enumerate(rows) if index % num_shards == shard_id]
    shard_manifest = manifest_dir / f"shard-{shard_id:03d}.jsonl"
    shard_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in shard_rows),
        encoding="utf-8",
    )
    for spec in model_specs:
        output_dir = output_root / "models" / spec["key"] / "shards" / f"shard-{shard_id:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        config_dir = config_root / spec["key"]
        config_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "manifest_path": str(shard_manifest),
            "output_path": str(output_dir / "metrics.json"),
            "predictions_path": str(output_dir / "predictions.jsonl"),
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "prediction_mode": "model",
            "checkpoint_path": spec["checkpoint_path"],
            "max_examples": 0,
        }
        (config_dir / f"shard-{shard_id:03d}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

metadata = {
    "eval_manifest_path": str(eval_manifest_path),
    "selected_manifest_path": str(selected_manifest_path),
    "selected_examples": len(rows),
    "num_shards": num_shards,
    "model_specs_path": os.environ["MODEL_SPECS_PATH"],
    "generation": {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
    },
    "models": model_specs,
}
(output_root / "shard_plan.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(metadata, indent=2, sort_keys=True))
PY

echo "date=$(date -Is)"
echo "hostname=$(hostname -f)"
echo "git_commit=$(git rev-parse HEAD)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "PREVIOUS_SFT_CHECKPOINT_PATH=$PREVIOUS_SFT_CHECKPOINT_PATH"
echo "NEW_SFT_ROOT=$NEW_SFT_ROOT"
echo "CHECKPOINT_STEPS=$CHECKPOINT_STEPS"
echo "EVAL_MANIFEST_PATH=$EVAL_MANIFEST_PATH"
echo "EVAL_MAX_EXAMPLES=$EVAL_MAX_EXAMPLES"
echo "EVAL_MAX_NEW_TOKENS=$EVAL_MAX_NEW_TOKENS"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true
fi

run_model_shards() {
  local model_key="$1"
  echo "running_${model_key}_shards=1"
  srun \
    --kill-on-bad-exit=1 \
    --ntasks="$NUM_SHARDS" \
    --ntasks-per-node="$NUM_SHARDS" \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --export=ALL,MODEL_KEY="$model_key" \
    bash -lc '
      set -euo pipefail
      SHARD_ID="$(printf "%03d" "$SLURM_PROCID")"
      SHARD_CONFIG="$OUTPUT_ROOT/configs/$MODEL_KEY/shard-${SHARD_ID}.yaml"
      SHARD_LOG_DIR="$OUTPUT_ROOT/logs"
      {
        echo "date=$(date -Is)"
        echo "hostname=$(hostname -f)"
        echo "MODEL_KEY=$MODEL_KEY"
        echo "SLURM_PROCID=$SLURM_PROCID"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
        echo "SHARD_CONFIG=$SHARD_CONFIG"
        PYTHONPATH=src python scripts/eval/eval_local_holdout.py \
          --model-config "$MODEL_CONFIG" \
          --eval-config "$SHARD_CONFIG"
      } > "$SHARD_LOG_DIR/${MODEL_KEY}-shard-${SHARD_ID}.out" \
        2> "$SHARD_LOG_DIR/${MODEL_KEY}-shard-${SHARD_ID}.err"
    ' < /dev/null
  echo "running_${model_key}_shards=0"
}

python - <<'PY' > "$OUTPUT_ROOT/model_keys.txt"
import json
import os
from pathlib import Path
for spec in json.loads(Path(os.environ["MODEL_SPECS_PATH"]).read_text(encoding="utf-8")):
    print(spec["key"])
PY

while IFS= read -r model_key; do
  [[ -n "$model_key" ]] || continue
  run_model_shards "$model_key"
done < "$OUTPUT_ROOT/model_keys.txt"

python - <<'PY'
import json
import os
from pathlib import Path

from agri_vlm.data.manifest_io import read_manifest
from agri_vlm.evaluation.local_eval import score_local_predictions
from agri_vlm.evaluation.reporting import build_prediction_rows
from agri_vlm.utils.io import ensure_dir, write_json, write_jsonl

output_root = Path(os.environ["OUTPUT_ROOT"])
selected_manifest_path = Path(os.environ["SELECTED_MANIFEST_PATH"])
model_specs_path = Path(os.environ["MODEL_SPECS_PATH"])
num_shards = int(os.environ["NUM_SHARDS"])
rows = read_manifest(selected_manifest_path)
row_by_sample_id = {row.sample_id: row for row in rows}
if len(row_by_sample_id) != len(rows):
    raise SystemExit("Selected eval rows contain duplicate sample ids.")

model_specs = json.loads(model_specs_path.read_text(encoding="utf-8"))

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

for spec in model_specs:
    model_key = spec["key"]
    predictions_by_sample_id = {}
    for shard_id in range(num_shards):
        predictions_path = output_root / "models" / model_key / "shards" / f"shard-{shard_id:03d}" / "predictions.jsonl"
        if not predictions_path.is_file():
            raise SystemExit(f"Missing shard predictions: {predictions_path}")
        for prediction_row in read_jsonl(predictions_path):
            sample_id = prediction_row["sample_id"]
            if sample_id in predictions_by_sample_id:
                raise SystemExit(f"Duplicate prediction for sample id {sample_id} in {model_key}")
            predictions_by_sample_id[sample_id] = prediction_row["prediction"]
    missing = sorted(set(row_by_sample_id) - set(predictions_by_sample_id))
    extra = sorted(set(predictions_by_sample_id) - set(row_by_sample_id))
    if missing or extra:
        raise SystemExit(
            "Prediction sample mismatch for %s: missing=%s extra=%s"
            % (model_key, missing[:10], extra[:10])
        )
    ordered_predictions = [predictions_by_sample_id[row.sample_id] for row in rows]
    output_dir = output_root / "models" / model_key / "local_holdout"
    ensure_dir(output_dir)
    metrics = score_local_predictions(rows, ordered_predictions)
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", build_prediction_rows(rows, ordered_predictions))
    spec["merged_predictions_path"] = str(output_dir / "predictions.jsonl")
    spec["merged_metrics_path"] = str(output_dir / "metrics.json")

model_specs_path.write_text(json.dumps(model_specs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"merged_models": model_specs}, indent=2, sort_keys=True))
PY

PYTHONPATH=src:benchmarks/vlm_baselines python - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

specs = json.loads(Path(os.environ["MODEL_SPECS_PATH"]).read_text(encoding="utf-8"))
args = [
    sys.executable,
    "scripts/eval/build_sft_round_comparison_report.py",
    "--manifest",
    os.environ["SELECTED_MANIFEST_PATH"],
]
for spec in specs:
    args.extend(
        [
            "--model",
            "%s|%s|%s|%s"
            % (
                spec["key"],
                spec["display_name"],
                spec["checkpoint_type"],
                spec["merged_predictions_path"],
            ),
        ]
    )
args.extend(
    [
        "--output-json",
        os.environ["METRICS_PATH"],
        "--output-jsonl",
        os.environ["COMPARISON_PATH"],
        "--summary-md",
        os.environ["SUMMARY_PATH"],
        "--pairwise-md",
        os.environ["PAIRWISE_REPORT_PATH"],
        "--title",
        "New SFT Checkpoint Sweep vs Previous SFT",
        "--phase",
        "sft_eval",
        "--split",
        "selected",
        "--generation-label",
        os.environ["GENERATION_LABEL"],
        "--focus-model-key",
        specs[-1]["key"],
        "--compare-model-key",
        "previous_sft",
        "--max-summary-examples",
        "16",
    ]
)
subprocess.run(args, check=True)
PY

echo "sft_checkpoint_sweep_completed=1"
echo "summary_path=$SUMMARY_PATH"
echo "metrics_path=$METRICS_PATH"
echo "comparison_path=$COMPARISON_PATH"
echo "pairwise_report_path=$PAIRWISE_REPORT_PATH"
