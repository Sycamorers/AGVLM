#!/bin/bash -l
#SBATCH --job-name=agri-vlm-sft-infer-check
#SBATCH --partition=hpg-turin
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --time=04:00:00
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
export TMPDIR="${AGRI_VLM_TMPDIR:-$PWD/.tmp/sft-inference-check-${SLURM_JOB_ID:-manual}}"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TMPDIR/xdg-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/phi4_reasoning_vision_15b_turin_24g_4bit_rl.yaml}"
SFT_CHECKPOINT_PATH="${SFT_CHECKPOINT_PATH:-/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1}"
EVAL_MANIFEST_PATH="${EVAL_MANIFEST_PATH:-data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-512}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-96}"
EVAL_MIN_NEW_TOKENS="${EVAL_MIN_NEW_TOKENS:-2}"
EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-false}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_SHARDS="${NUM_SHARDS:-16}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference_checks/$RUN_TAG}"
BASE_OUTPUT_DIR="$OUTPUT_ROOT/base"
SFT_OUTPUT_DIR="$OUTPUT_ROOT/sft"
COMPARISON_PATH="$OUTPUT_ROOT/comparison.jsonl"
SUMMARY_PATH="$OUTPUT_ROOT/summary.md"
PAIRWISE_REPORT_PATH="$OUTPUT_ROOT/pairwise_comparison.md"
export MODEL_CONFIG SFT_CHECKPOINT_PATH EVAL_MANIFEST_PATH EVAL_MAX_EXAMPLES
export EVAL_MAX_NEW_TOKENS EVAL_MIN_NEW_TOKENS EVAL_DO_SAMPLE EVAL_BATCH_SIZE NUM_SHARDS OUTPUT_ROOT
export BASE_OUTPUT_DIR SFT_OUTPUT_DIR COMPARISON_PATH SUMMARY_PATH PAIRWISE_REPORT_PATH

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
[[ -d "$SFT_CHECKPOINT_PATH" ]] || fail_fast "SFT checkpoint path missing: $SFT_CHECKPOINT_PATH"
[[ "$NUM_SHARDS" -le 16 ]] || fail_fast "NUM_SHARDS must be <=16; got $NUM_SHARDS"
[[ "$NUM_SHARDS" -gt 0 ]] || fail_fast "NUM_SHARDS must be positive"

python - "$SFT_CHECKPOINT_PATH" <<'PY'
import json
from pathlib import Path
import sys

from safetensors import safe_open

checkpoint = Path(sys.argv[1])
adapter_config_path = checkpoint / "adapter_config.json"
adapter_model_path = checkpoint / "adapter_model.safetensors"
if not adapter_config_path.is_file():
    raise SystemExit(f"adapter_config.json missing under {checkpoint}")
if not adapter_model_path.is_file():
    raise SystemExit(f"adapter_model.safetensors missing under {checkpoint}")

adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
with safe_open(adapter_model_path, framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    if not keys:
        raise SystemExit(f"adapter_model.safetensors has no tensors: {adapter_model_path}")
    first_tensor = handle.get_tensor(keys[0])
    validation = {
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
    }
print(json.dumps(validation, indent=2, sort_keys=True))
PY

python - <<'PY'
import json
import os
from pathlib import Path

import yaml

eval_manifest_path = Path(os.environ["EVAL_MANIFEST_PATH"])
output_root = Path(os.environ["OUTPUT_ROOT"])
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
sft_checkpoint_path = os.environ["SFT_CHECKPOINT_PATH"]

rows = [
    json.loads(line)
    for line in eval_manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if max_examples > 0:
    rows = rows[:max_examples]
if not rows:
    raise SystemExit(f"No evaluation rows selected from {eval_manifest_path}")

sample_ids = [str(row.get("sample_id")) for row in rows]
duplicates = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
if duplicates:
    raise SystemExit(f"Duplicate sample ids in selected eval rows: {duplicates[:10]}")

manifest_dir = output_root / "manifests"
config_root = output_root / "configs"
manifest_dir.mkdir(parents=True, exist_ok=True)
config_root.mkdir(parents=True, exist_ok=True)

for shard_id in range(num_shards):
    shard_rows = [row for index, row in enumerate(rows) if index % num_shards == shard_id]
    shard_manifest = manifest_dir / f"shard-{shard_id:03d}.jsonl"
    shard_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in shard_rows),
        encoding="utf-8",
    )
    for model_label, checkpoint_path in {
        "base": None,
        "sft": sft_checkpoint_path,
    }.items():
        output_dir = output_root / model_label / "shards" / f"shard-{shard_id:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        config_dir = config_root / model_label
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
            "checkpoint_path": checkpoint_path,
            "max_examples": 0,
        }
        (config_dir / f"shard-{shard_id:03d}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

metadata = {
    "eval_manifest_path": str(eval_manifest_path),
    "selected_examples": len(rows),
    "num_shards": num_shards,
    "generation": {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
    },
    "examples_per_shard": {
        f"shard-{shard_id:03d}": len(
            [row for index, row in enumerate(rows) if index % num_shards == shard_id]
        )
        for shard_id in range(num_shards)
    },
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
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "SFT_CHECKPOINT_PATH=$SFT_CHECKPOINT_PATH"
echo "EVAL_MANIFEST_PATH=$EVAL_MANIFEST_PATH"
echo "EVAL_MAX_EXAMPLES=$EVAL_MAX_EXAMPLES"
echo "EVAL_MAX_NEW_TOKENS=$EVAL_MAX_NEW_TOKENS"
echo "EVAL_MIN_NEW_TOKENS=$EVAL_MIN_NEW_TOKENS"
echo "EVAL_DO_SAMPLE=$EVAL_DO_SAMPLE"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true
fi

run_model_shards() {
  local model_label="$1"
  echo "running_${model_label}_shards=1"
  srun \
    --kill-on-bad-exit=1 \
    --ntasks="$NUM_SHARDS" \
    --ntasks-per-node=2 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --export=ALL,MODEL_LABEL="$model_label" \
    bash -lc '
      set -euo pipefail
      SHARD_ID="$(printf "%03d" "$SLURM_PROCID")"
      SHARD_CONFIG="$OUTPUT_ROOT/configs/$MODEL_LABEL/shard-${SHARD_ID}.yaml"
      SHARD_LOG_DIR="$OUTPUT_ROOT/logs"
      {
        echo "date=$(date -Is)"
        echo "hostname=$(hostname -f)"
        echo "MODEL_LABEL=$MODEL_LABEL"
        echo "SLURM_PROCID=$SLURM_PROCID"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
        echo "SHARD_CONFIG=$SHARD_CONFIG"
        PYTHONPATH=src python scripts/eval/eval_local_holdout.py \
          --model-config "$MODEL_CONFIG" \
          --eval-config "$SHARD_CONFIG"
      } > "$SHARD_LOG_DIR/${MODEL_LABEL}-shard-${SHARD_ID}.out" \
        2> "$SHARD_LOG_DIR/${MODEL_LABEL}-shard-${SHARD_ID}.err"
    '
  echo "running_${model_label}_shards=0"
}

run_model_shards base
run_model_shards sft

python - <<'PY'
import json
import os
from pathlib import Path

from agri_vlm.data.manifest_io import read_manifest
from agri_vlm.evaluation.local_eval import score_local_predictions
from agri_vlm.evaluation.reporting import build_prediction_rows
from agri_vlm.utils.io import ensure_dir, write_json, write_jsonl

output_root = Path(os.environ["OUTPUT_ROOT"])
eval_manifest_path = Path(os.environ["EVAL_MANIFEST_PATH"])
max_examples = int(os.environ["EVAL_MAX_EXAMPLES"])
num_shards = int(os.environ["NUM_SHARDS"])
base_output_dir = Path(os.environ["BASE_OUTPUT_DIR"])
sft_output_dir = Path(os.environ["SFT_OUTPUT_DIR"])
comparison_path = Path(os.environ["COMPARISON_PATH"])
summary_path = Path(os.environ["SUMMARY_PATH"])
pairwise_report_path = Path(os.environ["PAIRWISE_REPORT_PATH"])
generation_settings = {
    "max_new_tokens": int(os.environ["EVAL_MAX_NEW_TOKENS"]),
    "min_new_tokens": int(os.environ["EVAL_MIN_NEW_TOKENS"]),
    "do_sample": os.environ["EVAL_DO_SAMPLE"].strip().lower() in {"1", "true", "yes", "on"},
}

rows = read_manifest(eval_manifest_path)
if max_examples > 0:
    rows = rows[:max_examples]
row_by_sample_id = {row.sample_id: row for row in rows}
if len(row_by_sample_id) != len(rows):
    raise SystemExit("Selected eval rows contain duplicate sample ids.")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_predictions(model_label: str):
    predictions_by_sample_id = {}
    for shard_id in range(num_shards):
        predictions_path = (
            output_root
            / model_label
            / "shards"
            / f"shard-{shard_id:03d}"
            / "predictions.jsonl"
        )
        if not predictions_path.is_file():
            raise SystemExit(f"Missing shard predictions: {predictions_path}")
        for prediction_row in read_jsonl(predictions_path):
            sample_id = prediction_row["sample_id"]
            if sample_id in predictions_by_sample_id:
                raise SystemExit(f"Duplicate prediction for sample id {sample_id}")
            predictions_by_sample_id[sample_id] = prediction_row["prediction"]
    missing = sorted(set(row_by_sample_id) - set(predictions_by_sample_id))
    extra = sorted(set(predictions_by_sample_id) - set(row_by_sample_id))
    if missing or extra:
        raise SystemExit(
            "Prediction sample mismatch for %s: missing=%s extra=%s"
            % (model_label, missing[:10], extra[:10])
        )
    return [predictions_by_sample_id[row.sample_id] for row in rows]


def extract_user_text_prompt(row):
    prompt_parts = []
    for message in row.messages:
        if message.role != "user":
            continue
        for content_item in message.content:
            text = getattr(content_item, "text", None)
            if text:
                prompt_parts.append(text.strip())
    return "\n".join(part for part in prompt_parts if part)


base_predictions = collect_predictions("base")
sft_predictions = collect_predictions("sft")
base_metrics = score_local_predictions(rows, base_predictions)
sft_metrics = score_local_predictions(rows, sft_predictions)

ensure_dir(base_output_dir / "local_holdout")
ensure_dir(sft_output_dir / "local_holdout")
write_json(base_output_dir / "local_holdout" / "metrics.json", base_metrics)
write_json(sft_output_dir / "local_holdout" / "metrics.json", sft_metrics)
write_jsonl(
    base_output_dir / "local_holdout" / "predictions.jsonl",
    build_prediction_rows(rows, base_predictions),
)
write_jsonl(
    sft_output_dir / "local_holdout" / "predictions.jsonl",
    build_prediction_rows(rows, sft_predictions),
)

comparison_rows = []
changed = 0
base_empty = 0
sft_empty = 0
for row, base_prediction, sft_prediction in zip(rows, base_predictions, sft_predictions):
    prediction_changed = base_prediction != sft_prediction
    changed += int(prediction_changed)
    base_empty += int(not base_prediction.strip())
    sft_empty += int(not sft_prediction.strip())
    comparison_rows.append(
        {
            "task": "local_holdout",
            "sample_id": row.sample_id,
            "source_dataset": row.source_dataset,
            "task_type": row.task_type,
            "question_text": extract_user_text_prompt(row),
            "reference_text": row.target.answer_text or row.target.canonical_label or "",
            "base_prediction": base_prediction,
            "sft_prediction": sft_prediction,
            "prediction_changed": prediction_changed,
            "images": list(row.images),
        }
    )
write_jsonl(comparison_path, comparison_rows)

summary_lines = [
    "# SFT Inference Check",
    "",
    f"- Manifest: `{eval_manifest_path}`",
    f"- Examples: `{len(rows)}`",
    f"- Shards/GPUs: `{num_shards}`",
    f"- Generation: `do_sample={generation_settings['do_sample']}`, `min_new_tokens={generation_settings['min_new_tokens']}`, `max_new_tokens={generation_settings['max_new_tokens']}`",
    f"- Prediction changes: `{changed}/{len(rows)}`",
    f"- Empty base predictions: `{base_empty}/{len(rows)}`",
    f"- Empty SFT predictions: `{sft_empty}/{len(rows)}`",
    "",
    "| Metric | Base | SFT |",
    "| --- | ---: | ---: |",
]
for metric_name in sorted(set(base_metrics) | set(sft_metrics)):
    base_value = base_metrics.get(metric_name)
    sft_value = sft_metrics.get(metric_name)
    if isinstance(base_value, (int, float)) and isinstance(sft_value, (int, float)):
        summary_lines.append(f"| {metric_name} | {base_value:.6f} | {sft_value:.6f} |")
summary_lines.extend(
    [
        "",
        f"- Base predictions: `{base_output_dir / 'local_holdout' / 'predictions.jsonl'}`",
        f"- SFT predictions: `{sft_output_dir / 'local_holdout' / 'predictions.jsonl'}`",
        f"- Side-by-side JSONL: `{comparison_path}`",
        f"- Pairwise Markdown: `{pairwise_report_path}`",
        f"- Shard logs: `{output_root / 'logs'}`",
    ]
)
summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")


def md_cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "<br>".join(str(item) for item in value)
    else:
        value = str(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\\", "\\\\")
    value = value.replace("|", "\\|")
    value = value.replace("\n", "<br>")
    if value == "":
        return "_empty_"
    return value


pairwise_lines = [
    "# SFT vs Base Pairwise Inference Report",
    "",
    "## Run Summary",
    "",
    f"- Source comparison JSONL: `{comparison_path}`",
    f"- Rows: `{len(comparison_rows)}`",
    f"- Generation: `do_sample={generation_settings['do_sample']}`, `min_new_tokens={generation_settings['min_new_tokens']}`, `max_new_tokens={generation_settings['max_new_tokens']}`",
    f"- Prediction changes: `{changed}/{len(rows)}`",
    f"- Empty base predictions: `{base_empty}/{len(rows)}`",
    f"- Empty SFT predictions: `{sft_empty}/{len(rows)}`",
    "- Full report keeps predictions untruncated; line breaks inside model outputs are rendered as `<br>`.",
    "",
    "## Metrics",
    "",
    "| Metric | Base | SFT |",
    "| --- | ---: | ---: |",
]
for metric_name in sorted(set(base_metrics) | set(sft_metrics)):
    base_value = base_metrics.get(metric_name)
    sft_value = sft_metrics.get(metric_name)
    if isinstance(base_value, (int, float)) and isinstance(sft_value, (int, float)):
        pairwise_lines.append(f"| {metric_name} | {base_value:.6f} | {sft_value:.6f} |")
pairwise_lines.extend(
    [
        "",
        "## Pairwise Predictions",
        "",
        "| # | Dataset | Task Type | Sample ID | Image(s) | Question / Prompt | Reference | Base Prediction | SFT Prediction | Changed |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
)
for index, comparison_row in enumerate(comparison_rows, start=1):
    pairwise_lines.append(
        "| %d | %s | %s | `%s` | %s | %s | %s | %s | %s | %s |"
        % (
            index,
            md_cell(comparison_row["source_dataset"]),
            md_cell(comparison_row["task_type"]),
            md_cell(comparison_row["sample_id"]),
            md_cell(comparison_row["images"]),
            md_cell(comparison_row["question_text"]),
            md_cell(comparison_row["reference_text"]),
            md_cell(comparison_row["base_prediction"]),
            md_cell(comparison_row["sft_prediction"]),
            "yes" if comparison_row["prediction_changed"] else "no",
        )
    )
pairwise_report_path.write_text("\n".join(pairwise_lines).rstrip() + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "base": base_metrics,
            "sft": sft_metrics,
            "changed": changed,
            "base_empty": base_empty,
            "sft_empty": sft_empty,
            "generation": generation_settings,
        },
        indent=2,
        sort_keys=True,
    )
)
print(f"comparison_path={comparison_path}")
print(f"summary_path={summary_path}")
print(f"pairwise_report_path={pairwise_report_path}")
PY

echo "sft_inference_check_completed=1"
