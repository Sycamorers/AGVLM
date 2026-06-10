#!/bin/bash -l
#SBATCH --job-name=agvlm-banana-overfit
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${AGRI_VLM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"

module load conda
module load cuda/12.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

export PYTHONPATH=src
export AGRI_VLM_DATA_ROOT="${AGRI_VLM_DATA_ROOT:-$PWD/data}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export TMPDIR="${AGRI_VLM_TMPDIR:-$PWD/.tmp/micro-banana-overfit-${SLURM_JOB_ID:-manual}}"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TMPDIR/xdg-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export AGRI_VLM_CLASSIFICATION_PROMPT_FORMAT=label_only

DATA_CONFIG="${DATA_CONFIG:-configs/data/sft_micro_banana_overfit_label_only_phi4_max3.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/phi4_reasoning_vision_15b_b200.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/sft_phi4_reasoning_vision_15b_b200_1gpu_micro_banana_overfit_label_only.yaml}"
MODEL_KEY="${MODEL_KEY:-agvlm_phi4_sft_micro_banana_overfit_b200_1gpu_candidate}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/manifests/full/sft_micro_banana_overfit_label_only_train.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-data/manifests/full/sft_micro_banana_overfit_label_only_eval.jsonl}"
BENCH_ROOT="${BENCH_ROOT:-benchmarks/vlm_baselines/results/micro_banana_overfit_${SLURM_JOB_ID:-manual}}"
TRAIN_BENCH_DIR="$BENCH_ROOT/train_constrained"
EVAL_BENCH_DIR="$BENCH_ROOT/eval_constrained"
REPORT_PATH="${REPORT_PATH:-reports/micro_banana_overfit_diagnosis_${SLURM_JOB_ID:-manual}.md}"

mkdir -p logs/slurm "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE" "$TMPDIR" "$BENCH_ROOT" reports

config_value() {
  python - "$1" "$2" <<'PY'
import sys
import yaml
from pathlib import Path

payload = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
value = payload.get(sys.argv[2])
print("" if value is None else value)
PY
}

fail_fast() {
  echo "ERROR: $1" >&2
  exit 2
}

[[ -f "$DATA_CONFIG" ]] || fail_fast "data config missing: $DATA_CONFIG"
[[ -f "$MODEL_CONFIG" ]] || fail_fast "model config missing: $MODEL_CONFIG"
[[ -f "$TRAIN_CONFIG" ]] || fail_fast "train config missing: $TRAIN_CONFIG"

CHECKPOINT_DIR="$(config_value "$TRAIN_CONFIG" checkpoint_output_dir)"
RUN_DIR="$(config_value "$TRAIN_CONFIG" output_dir)"
export DATA_CONFIG MODEL_CONFIG TRAIN_CONFIG MODEL_KEY TRAIN_MANIFEST EVAL_MANIFEST
export BENCH_ROOT TRAIN_BENCH_DIR EVAL_BENCH_DIR REPORT_PATH CHECKPOINT_DIR RUN_DIR
[[ -n "$CHECKPOINT_DIR" ]] || fail_fast "checkpoint_output_dir missing from $TRAIN_CONFIG"
if [[ -e "$CHECKPOINT_DIR/adapter_config.json" || -e "$CHECKPOINT_DIR/adapter_model.safetensors" ]]; then
  fail_fast "target adapter already exists; refusing to overwrite: $CHECKPOINT_DIR"
fi

echo "date=$(date -Is)"
echo "hostname=$(hostname -f)"
echo "git_commit=$(git rev-parse HEAD)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_CONFIG=$DATA_CONFIG"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"
echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "BENCH_ROOT=$BENCH_ROOT"
echo "REPORT_PATH=$REPORT_PATH"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true
fi

echo "Preparing banana micro-overfit manifests"
python scripts/data/build_classification_probe_manifests.py --config "$DATA_CONFIG"

echo "Dry-running SFT config"
python scripts/train/train_sft.py \
  --model-config "$MODEL_CONFIG" \
  --train-config "$TRAIN_CONFIG" \
  --dry-run

echo "Training fresh LoRA micro-overfit adapter"
python scripts/train/train_sft.py \
  --model-config "$MODEL_CONFIG" \
  --train-config "$TRAIN_CONFIG"

[[ -f "$CHECKPOINT_DIR/adapter_config.json" ]] || fail_fast "adapter_config.json missing after training: $CHECKPOINT_DIR"
[[ -f "$CHECKPOINT_DIR/adapter_model.safetensors" ]] || fail_fast "adapter_model.safetensors missing after training: $CHECKPOINT_DIR"

echo "Benchmarking train split with constrained labels"
python benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split val \
  --manifest-path "$TRAIN_MANIFEST" \
  --split-dir benchmarks/vlm_baselines/splits_stage5_datafix \
  --batch-size 1 \
  --max-new-tokens 0 \
  --min-new-tokens 0 \
  --dtype bf16 \
  --quantization 4bit \
  --device cuda:0 \
  --output-dir "$TRAIN_BENCH_DIR" \
  --model-key "$MODEL_KEY" \
  --classification-decode-mode constrained

echo "Benchmarking heldout split with constrained labels"
python benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split test \
  --manifest-path "$EVAL_MANIFEST" \
  --split-dir benchmarks/vlm_baselines/splits_stage5_datafix \
  --batch-size 1 \
  --max-new-tokens 0 \
  --min-new-tokens 0 \
  --dtype bf16 \
  --quantization 4bit \
  --device cuda:0 \
  --output-dir "$EVAL_BENCH_DIR" \
  --model-key "$MODEL_KEY" \
  --classification-decode-mode constrained

python - <<'PY'
import json
import os
from pathlib import Path


def load_metric(root: Path) -> dict:
    candidates = sorted((root / "metrics").glob("*_metrics.json"))
    candidates = [path for path in candidates if not path.name.startswith("summary_table")]
    if len(candidates) != 1:
        raise SystemExit(f"Expected one metrics file under {root / 'metrics'}, found {candidates}")
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def pct(value):
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


train_root = Path(os.environ["TRAIN_BENCH_DIR"])
eval_root = Path(os.environ["EVAL_BENCH_DIR"])
report_path = Path(os.environ["REPORT_PATH"])
summary_path = Path("data/manifests/full/sft_micro_banana_overfit_label_only_summary.json")
train_metrics = load_metric(train_root)
eval_metrics = load_metric(eval_root)
manifest_summary = json.loads(summary_path.read_text(encoding="utf-8"))

train_cls = train_metrics.get("classification", {})
eval_cls = eval_metrics.get("classification", {})
lines = [
    "# Micro Banana Overfit Diagnosis",
    "",
    f"Date: {os.popen('date -Is').read().strip()}",
    "",
    "## Run",
    "",
    f"- Slurm job: `{os.environ.get('SLURM_JOB_ID', '')}`",
    f"- Train config: `{os.environ['TRAIN_CONFIG']}`",
    f"- Data config: `{os.environ['DATA_CONFIG']}`",
    f"- Adapter: `{os.environ['CHECKPOINT_DIR']}`",
    f"- Train benchmark: `{train_root}`",
    f"- Heldout benchmark: `{eval_root}`",
    "",
    "## Data",
    "",
    f"- Train rows: `{manifest_summary.get('train_output_rows')}`",
    f"- Heldout rows: `{manifest_summary.get('eval_output_rows')}`",
    f"- Selected labels: `{manifest_summary.get('selected_labels_by_source', {}).get('banana_disease')}`",
    "",
    "## Metrics",
    "",
    "| split | examples | top1 | macro F1 | weighted F1 | balanced acc | OOS |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    (
        f"| train | {train_cls.get('num_examples')} | {pct(train_cls.get('top1_accuracy'))} | "
        f"{pct(train_cls.get('macro_f1'))} | {pct(train_cls.get('weighted_f1'))} | "
        f"{pct(train_cls.get('balanced_accuracy'))} | {pct(train_cls.get('out_of_label_space_rate'))} |"
    ),
    (
        f"| heldout | {eval_cls.get('num_examples')} | {pct(eval_cls.get('top1_accuracy'))} | "
        f"{pct(eval_cls.get('macro_f1'))} | {pct(eval_cls.get('weighted_f1'))} | "
        f"{pct(eval_cls.get('balanced_accuracy'))} | {pct(eval_cls.get('out_of_label_space_rate'))} |"
    ),
    "",
    "## Interpretation",
    "",
]
if float(train_cls.get("top1_accuracy") or 0.0) >= 0.95:
    lines.extend(
        [
            "The fresh LoRA can memorize the tiny banana train split. The basic training/collator/adapter path is functional.",
            "The broader Stage7 failure should therefore be treated as a data/label-space/task-interference problem.",
        ]
    )
else:
    lines.extend(
        [
            "The fresh LoRA did not memorize the tiny banana train split.",
            "Before another broad SFT run, debug image loading, assistant-label masking, adapter attachment, and Phi-4 vision gradient flow.",
        ]
    )
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report_path.read_text(encoding="utf-8"))
PY

echo "finished=$(date -Is)"
