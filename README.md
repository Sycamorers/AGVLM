# agri-vlm-v1

Research codebase for a ground-level RGB agriculture VLM focused on disease, pest, symptom, consultation, and clarify-vs-respond tasks.

## Project Operating Docs

- [Project overview](docs/project_overview.md)
- [Research plan](docs/research_plan.md)
- [Experiment roadmap](docs/experiment_roadmap.md)
- [Progress tracker](docs/progress_tracker.md)
- [Training monitoring](docs/training_monitoring.md)
- [Benchmark plan](docs/benchmark_plan.md)
- [Results artifacts](docs/results_artifacts.md)
- [Paper outline](docs/paper_outline.md)
- [Session handoff](docs/session_handoff.md)

## Target Environment

- Cluster: UF HiPerGator
- Modules: `module load conda` and `module load cuda/12.9.1`
- Python: `3.11`
- GPUs: NVIDIA B200 class
- Training: single-node multi-GPU `torchrun`
- Default precision: `bf16`
- Default base model: `Qwen/Qwen3-VL-4B-Instruct`

## Quick Setup

```bash
cd /blue/hmedeiros/qinruoyao/agvlm
module load conda
module load cuda/12.9.1
bash scripts/hpc/prepare_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1
```

Recommended environment variables:

```bash
export AGRI_VLM_DATA_ROOT="$PWD/data"
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$PWD/.tmp"
mkdir -p "$AGRI_VLM_DATA_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"
```

## Verify Environment

```bash
PYTHONPATH=src python scripts/verify_environment.py
```

This prints Python, torch, CUDA availability, GPU inventory, bf16 support, distributed state, and the dataset/cache environment variables.

## Data Preparation

Default behavior is a deterministic prefix download of the first 10% of each supported split. The output tag is `partial_10pct`.

Public datasets with automatic partial download:
- PlantVillage
- PlantDoc
- PlantVillageVQA
- MIRAGE

Manual datasets:
- IP102: manual drop-in
- AgBase resources: manual drop-in
- Agri-LLaVA / Agri-400K: manual drop-in

Prepare the 10% subset:

```bash
PYTHONPATH=src python scripts/data/download_public_datasets.py --download-mode partial --fraction 0.1
PYTHONPATH=src python scripts/data/normalize_all.py --download-mode partial --fraction 0.1
PYTHONPATH=src python scripts/data/build_sft_manifest.py --download-mode partial --fraction 0.1
PYTHONPATH=src python scripts/data/build_rl_manifest.py --download-mode partial --fraction 0.1
PYTHONPATH=src python scripts/data/build_eval_manifest.py --download-mode partial --fraction 0.1
PYTHONPATH=src python scripts/data/dataset_report.py --download-mode partial --fraction 0.1
```

Cluster wrapper:

```bash
DOWNLOAD_MODE=partial SAMPLE_FRACTION=0.1 bash scripts/hpc/run_data_prep.sh
```

Generated outputs:

- raw subsets: `data/raw/<dataset>/<subset_tag>/`
- normalized per-dataset manifests: `data/interim/<subset_tag>/`
- merged manifests: `data/manifests/<subset_tag>/`
- dataset report: `data/manifests/<subset_tag>/dataset_report.json` and `.md`

## Full Rerun Later

No code changes are required. Switch the data mode to full and rerun the same workflow.

```bash
PYTHONPATH=src python scripts/data/download_public_datasets.py --download-mode full --fraction 1.0
PYTHONPATH=src python scripts/data/normalize_all.py --download-mode full --fraction 1.0
PYTHONPATH=src python scripts/data/build_sft_manifest.py --download-mode full --fraction 1.0
PYTHONPATH=src python scripts/data/build_rl_manifest.py --download-mode full --fraction 1.0
PYTHONPATH=src python scripts/data/build_eval_manifest.py --download-mode full --fraction 1.0
PYTHONPATH=src python scripts/data/dataset_report.py --download-mode full --fraction 1.0
```

Equivalent Make targets:

```bash
make data-partial
make data-full
make data-report
```

## Training

Smoke pipeline:

```bash
bash scripts/run_smoke_test.sh
```

Multi-GPU SFT:

```bash
PYTHONPATH=src python scripts/launch_torchrun.py \
  --nproc-per-node 8 \
  scripts/train/train_sft.py -- \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_lora_full_b200_multigpu.yaml
```

Multi-GPU RL:

```bash
PYTHONPATH=src python scripts/launch_torchrun.py \
  --nproc-per-node 8 \
  scripts/train/train_rl_grpo.py -- \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/rl_grpo_b200_multigpu.yaml
```

Training outputs are standardized under each run directory:

- `resolved_config.yaml`
- `run_metadata.json`
- `metrics/train_metrics.jsonl`
- `metrics.jsonl`
- `tensorboard/`
- `artifact_manifest.json` after successful training

TensorBoard:

```bash
tensorboard --logdir outputs --host 0.0.0.0 --port 6006
```

Run `bash scripts/hpc/prepare_env.sh` once after updating the repo so the environment installs `tensorboard` and artifact plotting dependencies.

The current active milestone is the full L4 SFT rerun or resume. The latest visible L4 log shows CUDA OOM during fp32 loss conversion; the L4 configs now use `loss_chunk_size: 1024`.

## Evaluation

Single-run local holdout eval with model predictions:

```bash
PYTHONPATH=src python scripts/eval/eval_local_holdout.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --eval-config configs/eval/local_holdout_full.yaml \
  --prediction-mode model \
  --predictions-output outputs/eval/full/local_holdout_predictions.jsonl
```

Baseline or post-FT benchmark suite:

```bash
PYTHONPATH=src python scripts/eval/run_benchmark.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --tasks local_holdout mirage_mmst mirage_mmmt \
  --prediction-mode model \
  --output-dir outputs/benchmarks/base-qwen3-vl-4b
```

Run the same suite against an SFT checkpoint:

```bash
PYTHONPATH=src python scripts/eval/run_benchmark.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --tasks local_holdout mirage_mmst mirage_mmmt \
  --prediction-mode model \
  --checkpoint-path outputs/sft/qwen3-vl-4b-lora-full-b200 \
  --output-dir outputs/benchmarks/sft-qwen3-vl-4b-lora-full-b200
```

```bash
PYTHONPATH=src python scripts/eval/eval_local_holdout.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --eval-config configs/eval/local_holdout_full.yaml
```

MIRAGE and the local holdout use separate eval manifests under `data/manifests/full/`. The benchmark wrapper writes both aggregate metrics and per-example prediction JSONL files.

Check benchmark readiness:

```bash
PYTHONPATH=src python scripts/benchmarks/benchmark_status.py \
  --download-mode full \
  --fraction 1.0
```

Export benchmark tables:

```bash
PYTHONPATH=src python scripts/artifacts/export_benchmark_tables.py \
  --run Base-Qwen3-VL-4B outputs/benchmarks/base-qwen3-vl-4b_local_holdout_256
```

Export training curves:

```bash
PYTHONPATH=src python scripts/artifacts/export_training_artifacts.py \
  --run-dir outputs/sft/qwen3-vl-4b-lora-full-l4
```

## Repo Layout

```text
configs/        model, data, train, and eval configs
data/           raw subsets, normalized data, merged manifests
docs/           short project docs and decision log
scripts/        setup, HPC wrappers, data prep, train, and eval entrypoints
src/agri_vlm/   library code
tests/          unit tests and smoke checks
```

## Known Limitations

- Full SFT is still the active blocking milestone; the latest visible L4 job failed with CUDA OOM and needs a rerun or B200 fallback.
- IP102, AgBase resources, and Agri-LLaVA still require manual staging when rebuilding from scratch because this repo does not accept full-archive downloads just to keep 10%.
- AgMMU and AgroBench are tracked in the benchmark registry but still need verified access, normalizers, eval configs, and task entries.
- `flash-attn` remains optional until it is validated against the target CUDA 12.9.1 image.

## TODO Summary

Top open items are tracked in [TODO.md](/blue/hmedeiros/qinruoyao/agvlm/TODO.md). Current P0 items are:

- rerun or resume full SFT after the L4 OOM
- run post-SFT local and MIRAGE benchmarks
- export SFT training curves and benchmark tables for the paper
