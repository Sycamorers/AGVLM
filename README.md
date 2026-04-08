# agri-vlm-v1

Research codebase for an agriculture-specialized vision-language model focused on ground-level RGB disease, pest, symptom, and consultation tasks.

## Target

- GPUs: NVIDIA B200 class
- CUDA toolkit assumption: 12.9.1
- Multi-GPU: single-node `torchrun` is the primary training path
- Python: 3.11
- Default precision: bf16
- Default base model: `Qwen/Qwen3-VL-4B-Instruct`

The repo supports:
- dataset normalization into one multimodal JSONL schema
- supervised fine-tuning
- GRPO post-training
- evaluation
- smoke validation

## Quick Setup

Create a virtual environment and install the CUDA 12.9 wheel stack:

```bash
PYTHON_BIN=python3.11 bash scripts/bootstrap_env.sh
```

The bootstrap script installs:
- PyTorch `2.8.0` from the official `cu129` wheel index
- editable project dependencies
- optional `qwen-vl-utils`

Optional:
- `INSTALL_FLASH_ATTN=1` installs `flash-attn`, but this path still needs validation on the target CUDA 12.9.1 / B200 image.

## Verify Environment

Single process:

```bash
PYTHONPATH=src python3.11 scripts/verify_environment.py
```

Distributed sanity check on a multi-GPU host:

```bash
PYTHONPATH=src python3.11 scripts/launch_torchrun.py \
  --nproc-per-node 8 \
  scripts/verify_environment.py
```

The verifier prints:
- Python version
- torch version
- CUDA availability
- CUDA version reported by torch
- `nvcc` / `nvidia-smi` details if available
- GPU names and count
- bf16 support
- distributed backend info when launched under `torchrun`

## Data Preparation

Create manual dataset slots:

```bash
PYTHONPATH=src python3.11 scripts/data/prepare_manual_dataset_slots.py
```

Place approved raw data under `data/raw/<dataset>/`, then run the matching normalizer:

```bash
PYTHONPATH=src python3.11 scripts/data/normalize_plantvillage.py
PYTHONPATH=src python3.11 scripts/data/normalize_plantdoc.py
PYTHONPATH=src python3.11 scripts/data/normalize_ip102.py
PYTHONPATH=src python3.11 scripts/data/normalize_plantvillage_vqa.py
PYTHONPATH=src python3.11 scripts/data/normalize_agbase.py
PYTHONPATH=src python3.11 scripts/data/normalize_mirage.py
PYTHONPATH=src python3.11 scripts/data/normalize_agrillava.py
PYTHONPATH=src python3.11 scripts/data/normalize_agmmu.py
```

Build manifests:

```bash
PYTHONPATH=src python3.11 scripts/data/build_sft_manifest.py --config configs/data/sft_build.yaml
PYTHONPATH=src python3.11 scripts/data/build_rl_manifest.py --config configs/data/rl_build.yaml
PYTHONPATH=src python3.11 scripts/data/build_eval_manifest.py --config configs/data/eval_build.yaml
```

## Training

Smoke validation:

```bash
bash scripts/run_smoke_test.sh
```

Single-process SFT:

```bash
PYTHONPATH=src python3.11 scripts/train/train_sft.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_lora.yaml
```

Multi-GPU SFT:

```bash
PYTHONPATH=src python3.11 scripts/launch_torchrun.py \
  --nproc-per-node 8 \
  scripts/train/train_sft.py -- \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_lora_b200_multigpu.yaml
```

Single-process RL:

```bash
PYTHONPATH=src python3.11 scripts/train/train_rl_grpo.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/rl_grpo_lora.yaml
```

Multi-GPU RL:

```bash
PYTHONPATH=src python3.11 scripts/launch_torchrun.py \
  --nproc-per-node 8 \
  scripts/train/train_rl_grpo.py -- \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/rl_grpo_b200_multigpu.yaml
```

Convenience targets:

```bash
make smoke
make sft-dist NPROC_PER_NODE=8
make rl-dist NPROC_PER_NODE=8
make eval
```

## Evaluation

```bash
PYTHONPATH=src python3.11 scripts/eval/eval_local_holdout.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --eval-config configs/eval/local_holdout.yaml
```

## Repo Structure

```text
configs/        model, data, train, and eval configs
data/           raw slots, normalized data, and manifests
docs/           short project docs and decision log
scripts/        setup, launch, data prep, train, and eval entrypoints
src/agri_vlm/   library code
tests/          smoke and unit tests
```

## Known Limitations

- This repository assumes a CUDA 12.9.1-compatible system image, but the current host used for validation is not that target environment.
- Multi-GPU launch wiring is implemented, but full B200 runtime validation still needs to be performed on real hardware.
- `flash-attn` is optional and not enabled by default because the build path still needs explicit CUDA 12.9.1 validation.
- No Dockerfile is included yet.

## TODO Summary

Top open items are tracked in [TODO.md](/blue/hmedeiros/qinruoyao/agvlm/TODO.md).

Current P0 items:
- validate `flash-attn` on the target CUDA 12.9.1 / B200 image
- run a real multi-GPU SFT and RL checkpoint/resume test on B200 hardware
- validate distributed RL throughput and stability before enabling optional faster rollout paths
