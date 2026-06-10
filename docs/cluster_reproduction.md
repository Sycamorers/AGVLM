# Cluster Reproduction Guide

This guide is the handoff path for rebuilding AGVLM on a new cluster. V1 remains scoped to ground-level RGB agricultural consultation, disease/pest classification, VQA, and clarify-vs-respond tasks.

## What Git Contains

The repository contains code, configs, Slurm wrappers, benchmark split manifests, reports, and documentation. It intentionally does not contain raw datasets, generated normalized manifests, model weights, checkpoints, Hugging Face caches, logs, or `outputs/` artifacts.

Ignored runtime paths:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `data/manifests/`
- `outputs/`
- `logs/`
- `.cache/`
- `.tmp/`

Manual or gated data must be staged on each cluster under the configured data root. The strict full-data commands below fail when these inputs are missing.

## Clone And Environment

```bash
git clone git@github.com:Sycamorers/AGVLM.git
cd AGVLM
git checkout main
```

Set cluster-local storage before installing or downloading anything:

```bash
export AGRI_VLM_REPO_ROOT="$PWD"
export AGRI_VLM_DATA_ROOT=/path/to/fast/agvlm/data
export HF_HOME=/path/to/fast/agvlm/huggingface
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TMPDIR=/path/to/fast/agvlm/tmp
export PYTHONPATH=src
mkdir -p "${AGRI_VLM_DATA_ROOT}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"
```

Most Slurm wrappers use `SLURM_SUBMIT_DIR` by default. Set
`AGRI_VLM_REPO_ROOT` when submitting from a directory other than the cloned
repository root.

Bootstrap a Python 3.11 environment:

```bash
PYTHON_BIN=python3.11 VENV_DIR=.venv bash scripts/bootstrap_env.sh
source .venv/bin/activate
PYTHONPATH=src python scripts/verify_environment.py
```

On clusters that require Conda modules, create an equivalent Python 3.11 env, install the same PyTorch wheel family used by `scripts/bootstrap_env.sh`, then run:

```bash
python -m pip install -e ".[dev,deepspeed]"
PYTHONPATH=src python scripts/verify_environment.py
```

Use `HF_TOKEN` only if the target cluster needs authenticated Hugging Face access:

```bash
export HF_TOKEN=...
```

## Data Acquisition

Create raw-data slots and download public sources:

```bash
PYTHONPATH=src python scripts/data/prepare_manual_dataset_slots.py --download-mode full --fraction 1.0
PYTHONPATH=src python scripts/data/download_public_datasets.py --download-mode full --fraction 1.0
```

Public automatic sources in `configs/data/datasets.yaml`:

- `plantvillage`
- `plantdoc`
- `rice_disease`
- `digigreen_crop_disease`
- `banana_disease`
- `tea_sickness`
- `plantvillage_vqa`
- `mirage`

Manual staging sources:

- `ip102`: official split files and images from the IP102 release.
- `agbase`: prepared AgBase/AgMMU-style consultation records and images.
- `agrillava`: Agri-LLaVA records and images; the repo does not rely on the broken Hub builder.

For each manual source, `prepare_manual_dataset_slots.py` writes:

```text
${AGRI_VLM_DATA_ROOT}/raw/<dataset>/full/README.manual.md
${AGRI_VLM_DATA_ROOT}/raw/<dataset>/full/MANIFEST.stub.json
```

Read those files, place the licensed/manual data in the corresponding raw directory, then rerun the strict rebuild. Do not commit raw/manual data to Git.

## Strict Full Manifest Rebuild

After public downloads and manual staging are complete:

```bash
make stage5-datafix-manifests PYTHON=python DATA_VALIDATE_WORKERS=16
make stage6-classification-probe-manifests PYTHON=python
make stage7-label-only-manifests PYTHON=python
make micro-banana-manifests PYTHON=python
```

`stage5-datafix-manifests` rebuilds the current full data path:

- `data/manifests/full/sft_manifest_stage5_datafix.jsonl`
- `data/manifests/full/sft_manifest_stage5_datafix.decode_valid_images.jsonl`
- `data/manifests/full/local_holdout_eval_stage5_datafix.decode_valid_images.jsonl`
- `data/manifests/full/sft_train_phi4_max3_stage5_no_eval_overlap.jsonl`
- `data/manifests/full/sft_eval_phi4_max3_stage5_raw_stratified1024.jsonl`
- `data/manifests/full/sft_train_phi4_max3_stage5_closed_label_datafix.jsonl`
- `data/manifests/full/sft_eval_phi4_max3_stage5_closed_label_stratified1024.jsonl`

The strict path uses `--fail-on-missing` for raw/interim sources. If any dataset is absent, fix the staged data instead of accepting a partial manifest.

## Smoke Validation

Smoke validation does not download full datasets or model weights:

```bash
bash scripts/run_smoke_test.sh
PYTHONPATH=src python -m pytest
```

Focused tests for the current classification-repair work:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_collators.py \
  tests/test_manifest_builders.py \
  tests/test_benchmark_model_adapters.py \
  tests/test_benchmark_phase_splits.py \
  tests/test_benchmark_prediction_parsing.py \
  tests/test_benchmark_run_baselines.py
```

## Training

The B200 SFT wrapper is config-driven. Rebuild manifests first, then submit the desired config.

Stage5 data-fix SFT:

```bash
sbatch \
  --job-name=agri-vlm-sft-stage5-datafix \
  --export=ALL,DATA_CONFIG=configs/data/sft_stage5_closed_label_datafix_phi4_max3.yaml,PREFLIGHT_CONFIG=configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage5_datafix_preflight.yaml,TRAIN_CONFIG=configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage5_datafix.yaml,TENSORBOARD_PORT=6015 \
  scripts/hpc/run_sft_b200_4gpu_phi4_reasoning_vision_15b_full_max3.slurm
```

Stage7 label-only classification SFT:

```bash
sbatch \
  --job-name=agri-vlm-sft-stage7-label-cls \
  --export=ALL,DATA_CONFIG=configs/data/sft_classification_only_stage7_label_only_phi4_max3.yaml,EXTRA_DATA_CONFIGS=configs/data/sft_classification_val_stage7_label_only_phi4_max3.yaml,PREFLIGHT_CONFIG=configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage7_label_only_classification_preflight.yaml,TRAIN_CONFIG=configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage7_label_only_classification.yaml,TENSORBOARD_PORT=6017 \
  scripts/hpc/run_sft_b200_4gpu_phi4_reasoning_vision_15b_full_max3.slurm
```

Stage7 mixed label-only SFT:

```bash
sbatch \
  --job-name=agri-vlm-sft-stage7-label-mixed \
  --export=ALL,DATA_CONFIG=configs/data/sft_classification_val_stage7_label_only_phi4_max3.yaml,PREFLIGHT_CONFIG=configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage7_label_only_mixed_preflight.yaml,TRAIN_CONFIG=configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage7_label_only_mixed.yaml,TENSORBOARD_PORT=6018 \
  scripts/hpc/run_sft_b200_4gpu_phi4_reasoning_vision_15b_full_max3.slurm
```

Checkpoint artifacts are written to the `checkpoint_output_dir` configured in each train YAML. On a new cluster, change those YAML values or bind-mount equivalent storage if `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/` is unavailable.

## Benchmarking

Frozen Stage5 benchmark splits are committed under:

```text
benchmarks/vlm_baselines/splits_stage5_datafix/
```

Run a dry validation first:

```bash
PYTHONPATH=benchmarks/vlm_baselines python benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split test \
  --split-dir benchmarks/vlm_baselines/splits_stage5_datafix \
  --model-key agvlm_phi4_sft_stage7_label_only_classification_b200_candidate \
  --max-samples 2 \
  --dry-run
```

For label-only classification prompts:

```bash
export AGRI_VLM_CLASSIFICATION_PROMPT_FORMAT=label_only
```

For closed-label constrained decoding during classification benchmarks:

```bash
export AGRI_VLM_CLASSIFICATION_DECODE_MODE=constrained
```

Then run the benchmark Slurm wrapper or call `run_baselines.py` without `--dry-run`.

## Generic Slurm Data Prep

For the older generic data path, the Slurm data-prep wrapper supports strict missing-input checks:

```bash
sbatch \
  --export=ALL,ENV_NAME=agri-vlm-v1,DOWNLOAD_MODE=full,SAMPLE_FRACTION=1.0,FAIL_ON_MISSING=1 \
  scripts/hpc/run_data_prep.slurm
```

Use the Makefile targets above for the current Stage5/Stage6/Stage7 research path.
