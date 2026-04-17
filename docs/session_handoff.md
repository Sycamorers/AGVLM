# Session Handoff

Last updated: 2026-04-17

## Current State

The repo is ready for the first real full-data SFT run.

Completed and normalized full datasets:

- `plantvillage`
  - raw rows: `54381`
  - interim: `data/interim/full/plantvillage.jsonl`
- `plantdoc`
  - raw rows: `2578`
  - interim: `data/interim/full/plantdoc.jsonl`
- `plantvillage_vqa`
  - raw rows: `193609`
  - interim: `data/interim/full/plantvillage_vqa.jsonl`
- `ip102`
  - normalized rows: `75222`
  - interim: `data/interim/full/ip102.jsonl`
- `agbase`
  - staged rows: `44849`
  - skipped rows without images: `247`
  - interim: `data/interim/full/agbase.jsonl`
- `mirage`
  - normalized rows: `40889`
  - interim: `data/interim/full/mirage.jsonl`
- `agrillava`
  - normalized rows: `1839`
  - interim: `data/interim/full/agrillava.jsonl`

Training manifests built from the finalized full normalized set:

- SFT manifest: `data/manifests/full/sft_manifest.jsonl`
  - rows: `327158`
- RL manifest: `data/manifests/full/rl_manifest.jsonl`
  - rows: `305978`

Updated dataset report:

- `data/manifests/full/dataset_report.json`
- `data/manifests/full/dataset_report.md`

Baseline-vs-fine-tuned evaluation workflow is now wired:

- `scripts/eval/run_benchmark.py`
  - runs `local_holdout`, `mirage_mmst`, and `mirage_mmmt`
  - supports the base model or an SFT checkpoint through `--checkpoint-path`
  - writes aggregate metrics and per-example predictions under a chosen output directory
- full eval configs:
  - `configs/eval/local_holdout_full.yaml`
  - `configs/eval/mirage_mmst_full.yaml`
  - `configs/eval/mirage_mmmt_full.yaml`
- full SFT config for later training:
  - `configs/train/sft_lora_full_b200_multigpu.yaml`

Baseline inference has been exercised on real B200 hardware:

- smoke run:
  - output dir: `outputs/benchmarks/base-qwen3-vl-4b_smoke`
  - examples: `4`
  - result: all-zero metrics under the current exact/normalized evaluator
- larger baseline slice:
  - output dir: `outputs/benchmarks/base-qwen3-vl-4b_local_holdout_256`
  - examples: `256`
  - metrics:
    - `label_accuracy: 0.0`
    - `label_macro_f1: 0.0`
    - `answer_exact_match: 0.0`
    - `clarify_accuracy: 0.0`
    - `average_reward: 0.0`
  - interpretation: inference is functioning, but the base model answers in verbose free text and does not match the current exact/normalized scoring assumptions.

## Remaining Blockers

There is no code blocker for SFT launch. The only observed blocker was cluster scheduling:

- a `4x B200` interactive request stayed pending with `QOSGrpCpuLimit`
- the next session should simply re-request GPUs and start training when the scheduler allows it

## What Changed In Code

- `pyproject.toml`
  - fixed the invalid `transformers>=5.5.0,<5.6.0` pin to `transformers>=4.56.1,<5`
- `src/agri_vlm/data/hf_download.py`
  - fixed `plantvillage_vqa` archive-backed download
  - fixed `mirage` split handling across configs
  - made image saves resumable by reusing existing non-empty files
- `src/agri_vlm/data/normalizers.py`
  - normalized MIRAGE decision tokens like `<Respond>` and `<Clarify>`
- `src/agri_vlm/data/conversation_format.py`
  - stops serializing `None` multimodal fields into Qwen chat messages
- `src/agri_vlm/modeling/model_factory.py`
  - falls back from `flash_attention_2` to `sdpa` when `flash_attn` is unavailable
- `src/agri_vlm/evaluation/inference.py`
  - supports batched inference and checkpoint-aware model loading
- `src/agri_vlm/evaluation/local_eval.py`
  - can return both metrics and per-example predictions
- `src/agri_vlm/evaluation/mirage_eval.py`
  - can return both metrics and per-example predictions
- `src/agri_vlm/evaluation/reporting.py`
  - serializes per-example prediction rows for benchmark artifacts
- `scripts/eval/run_benchmark.py`
  - runs reproducible pre-FT / post-FT benchmark suites
- `scripts/data/stage_manual_sources.py`
  - stages `ip102` from Google Drive
  - stages `agrillava` from Hub JSON + `Img.rar`
  - stages `agbase` from AgMMU fine-tuning JSON + multi-part archive
  - fixes `ip102` split-file rewriting to `ip102_v1.1/images/...`
  - fixes `agbase` image-path resolution when the extracted archive omits the `images_ft/` prefix

## Repro Commands

Environment:

```bash
cd /blue/hmedeiros/qinruoyao/agvlm
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

export AGRI_VLM_DATA_ROOT="$PWD/data"
export HF_HOME="$PWD/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$PWD/.tmp"
```

GPU shell for real inference:

```bash
srun -p hpg-b200 --gpus=1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash -l
module load conda
module load cuda/12.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

export AGRI_VLM_DATA_ROOT="$PWD/data"
export HF_HOME="$PWD/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$PWD/.tmp"
```

Baseline inference before fine-tuning:

```bash
cd /blue/hmedeiros/qinruoyao/agvlm
PYTHONPATH=src python scripts/eval/run_benchmark.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --tasks local_holdout \
  --prediction-mode model \
  --output-dir outputs/benchmarks/base-qwen3-vl-4b
```

Post-SFT inference on the same eval set:

```bash
cd /blue/hmedeiros/qinruoyao/agvlm
PYTHONPATH=src python scripts/eval/run_benchmark.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --tasks local_holdout \
  --prediction-mode model \
  --checkpoint-path outputs/sft/qwen3-vl-4b-lora-full-b200 \
  --output-dir outputs/benchmarks/sft-qwen3-vl-4b-lora-full-b200
```

Recommended next SFT launch on 4 GPUs:

```bash
cd /blue/hmedeiros/qinruoyao/agvlm
module load conda
module load cuda/12.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

export AGRI_VLM_DATA_ROOT="$PWD/data"
export HF_HOME="$PWD/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$PWD/.tmp"

PYTHONPATH=src python scripts/launch_torchrun.py \
  --nproc-per-node 4 \
  scripts/train/train_sft.py -- \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_lora_full_b200_multigpu.yaml
```

Requested but not allocated during this session:

```bash
srun -p hpg-b200 --gpus=4 --cpus-per-task=32 --mem=96G --time=23:00:00 --pty bash -l
```

If the training manifests need to be rebuilt:

```bash
PYTHONPATH=src python scripts/data/build_sft_manifest.py --download-mode full --fraction 1.0
PYTHONPATH=src python scripts/data/build_rl_manifest.py --download-mode full --fraction 1.0
PYTHONPATH=src python scripts/data/dataset_report.py --download-mode full --fraction 1.0
```

## Quick Checks

```bash
ls -lh data/interim/full/*.jsonl
ls -lh data/manifests/full/sft_manifest.jsonl
ls -lh data/manifests/full/rl_manifest.jsonl
sed -n '1,120p' data/manifests/full/dataset_report.md
```

## Files Worth Inspecting

- `pyproject.toml`
- `src/agri_vlm/data/hf_download.py`
- `src/agri_vlm/data/normalizers.py`
- `scripts/data/stage_manual_sources.py`
- `docs/session_handoff.md`
