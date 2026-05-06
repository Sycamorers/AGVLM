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

## Current Status

As of May 6, 2026, the active milestone is Llama 4 Scout full-data SFT on the max-3-image manifest using 4x B200 GPUs.

The completed adapter to keep and use as the next-stage base is:

```text
/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft
```

The AGBASE-disjoint continuation was cancelled at `global_step=500` because inline distributed generation evaluation stalled the job after loss eval. It did not write `checkpoint-500`, and the degraded `checkpoint-450` path was removed during cleanup. Do not resume from the disjoint run.

Local `outputs/` and `logs/` were intentionally cleaned; Orange SFT storage now keeps only the retained balanced adapter.

## Target Environment

- Cluster: UF HiPerGator
- Modules: `module load conda` and `module load cuda/12.9.1`
- Python: `3.11`
- GPUs: NVIDIA B200 class
- Training: single-node multi-GPU `torchrun`
- Default precision: `bf16`
- Active base model: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- Legacy smoke/base-eval model: `Qwen/Qwen3-VL-4B-Instruct`

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
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR="$PWD/.tmp"
mkdir -p "$AGRI_VLM_DATA_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"
```

## Verify Environment

```bash
PYTHONPATH=src python scripts/verify_environment.py
```

This prints Python, torch, CUDA availability, GPU inventory, bf16 support, distributed state, model access checks, and the dataset/cache environment variables.

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

Full-data rebuild:

```bash
make data-full
make data-report
```

The active Llama 4 training split is built from decode/aspect-valid manifests and removes eval image-group overlap:

```bash
PYTHONPATH=src python scripts/data/build_sft_train_eval_manifests.py \
  --config configs/data/sft_train_eval_llama4_max3.yaml
```

Generated Llama 4 max3 manifests:

- `data/manifests/full/sft_train_llama4_max3_no_eval_overlap.jsonl`
- `data/manifests/full/sft_eval_llama4_max3_stratified512.jsonl`
- `data/manifests/full/sft_train_eval_llama4_max3_summary.json`

## Training

Smoke pipeline:

```bash
bash scripts/run_smoke_test.sh
```

After HPG maintenance, run the 100-step B200 probe first:

```bash
sbatch \
  --export=ALL,TRAIN_CONFIG=configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml \
  scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm
```

If the probe completes with no OOM, acceptable step time, and valid checkpoint writes, launch the full max3 run:

```bash
sbatch scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm
```

Active training config:

- probe: `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml`
- full: `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced.yaml`
- Slurm wrapper: `scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm`
- model config: `configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml`
- DeepSpeed config: `configs/deepspeed/zero3_lora_b200_no_offload.json`

Do not enable inline generation metrics during large SFT runs. Keep `eval_generation_metrics: false` and run generation evaluation separately on selected checkpoints.

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

## Evaluation

Single-run local holdout eval with model predictions:

```bash
PYTHONPATH=src python scripts/eval/eval_local_holdout.py \
  --model-config configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml \
  --eval-config configs/eval/local_holdout_full.yaml \
  --prediction-mode model \
  --checkpoint-path <checkpoint_or_adapter_dir> \
  --predictions-output outputs/eval/full/local_holdout_predictions.jsonl
```

Post-SFT benchmark suite:

```bash
PYTHONPATH=src python scripts/eval/run_benchmark.py \
  --model-config configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml \
  --tasks local_holdout mirage_mmst mirage_mmmt \
  --prediction-mode model \
  --checkpoint-path <checkpoint_or_adapter_dir> \
  --output-dir outputs/benchmarks/<model_or_run_name>
```

MIRAGE and the local holdout use separate eval manifests under `data/manifests/full/`. The benchmark wrapper writes aggregate metrics and per-example prediction JSONL files.

Check benchmark readiness:

```bash
PYTHONPATH=src python scripts/benchmarks/benchmark_status.py \
  --download-mode full \
  --fraction 1.0
```

Export benchmark tables:

```bash
PYTHONPATH=src python scripts/artifacts/export_benchmark_tables.py \
  --run Llama4-SFT outputs/benchmarks/<model_or_run_name>
```

Export training curves:

```bash
PYTHONPATH=src python scripts/artifacts/export_training_artifacts.py \
  --run-dir outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced
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

- The full max3 Llama 4 Scout SFT run has not completed yet; the next action is the 100-step B200 probe after maintenance.
- IP102, AgBase resources, and Agri-LLaVA still require manual staging when rebuilding from scratch because this repo does not accept full-archive downloads just to keep 10%.
- AGBASE-only/disjoint continuation degraded validation behavior and should be treated as a debugging path, not the next-stage base.
- AgMMU and AgroBench are tracked in the benchmark registry but still need verified access, normalizers, eval configs, and task entries.
- `flash-attn` remains optional until it is validated against the target CUDA 12.9.1 image.

## TODO Summary

Top open items are tracked in [TODO.md](/blue/hmedeiros/qinruoyao/agvlm/TODO.md). Current P0 items are:

- submit and validate the 100-step B200 max3 probe from the retained balanced adapter
- launch the full max3 B200 run if the probe is healthy
- run separate generation evaluation and benchmark export after a completed full checkpoint exists
