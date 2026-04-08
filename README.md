# agri-vlm-v1

`agri-vlm-v1` is a runnable V1 research codebase for an agriculture-specialized vision-language model pipeline focused on ground-level RGB consultation tasks:
- plant disease identification
- insect and pest identification
- symptom explanation
- expert-style agricultural consultation
- clarify-vs-respond behavior for underspecified user questions

The repository includes:
- dataset slot preparation and normalization into one multimodal JSONL schema
- config-driven supervised fine-tuning
- config-driven GRPO post-training
- evaluation for AgMMU, MIRAGE, and a local holdout
- unit tests and a synthetic smoke pipeline

## Base model decision

As of April 8, 2026, the default base model is `Qwen/Qwen3-VL-4B-Instruct`.

Why this model:
- I could verify official public open-weight local checkpoints for `Qwen/Qwen3-VL-4B-Instruct` and `Qwen/Qwen3-VL-8B-Instruct` on the official Qwen Hugging Face organization.
- I could verify current official Hugging Face Transformers documentation for `Qwen3-VL` in the v5.5.0 docs.
- I could not verify any official public open-weight locally fine-tunable `Qwen3.6` vision-language checkpoint on the official Qwen Hugging Face org or the current Transformers docs. That absence is an inference from official sources, not a direct vendor statement.
- `4B` is the smallest practical dense checkpoint for V1 LoRA/QLoRA experimentation.
- `8B` is included as an optional second config for larger runs.

Verified source links:
- Transformers Qwen3-VL docs: https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl
- Qwen3-VL-4B-Instruct model card: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Qwen3-VL-8B-Instruct model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- TRL GRPO docs: https://huggingface.co/docs/trl/grpo_trainer

More detail is recorded in [`docs/decision_log.md`](docs/decision_log.md).

## Environment

Documented training environment:
- Python `3.10+`
- CUDA-capable GPU for real SFT/RL runs
- PyTorch `2.6+`
- Transformers `5.5.x`
- TRL `0.25.x`

Local setup:

```bash
PYTHON_BIN=python3.10 bash scripts/bootstrap_env.sh
```

Verify the environment:

```bash
PYTHONPATH=src python3 scripts/verify_environment.py
```

## Expected repository layout

```text
agri-vlm-v1/
  AGENTS.md
  README.md
  pyproject.toml
  Makefile
  configs/
  data/
  docs/
  scripts/
  src/agri_vlm/
  tests/
```

## Dataset preparation flow

This repo does not silently skip ambiguous, gated, or license-sensitive datasets. The default preparation flow creates explicit manual slots first.

1. Prepare manual slots and synthetic smoke data:

```bash
PYTHONPATH=src python3 scripts/data/prepare_manual_dataset_slots.py --with-smoke-data
```

2. Place real raw datasets into the slot directories under `data/raw/`.

3. Normalize each dataset into the unified JSONL schema:

```bash
PYTHONPATH=src python3 scripts/data/normalize_plantvillage.py
PYTHONPATH=src python3 scripts/data/normalize_plantdoc.py
PYTHONPATH=src python3 scripts/data/normalize_ip102.py
PYTHONPATH=src python3 scripts/data/normalize_plantvillage_vqa.py
PYTHONPATH=src python3 scripts/data/normalize_agbase.py
PYTHONPATH=src python3 scripts/data/normalize_mirage.py
PYTHONPATH=src python3 scripts/data/normalize_agrillava.py
PYTHONPATH=src python3 scripts/data/normalize_agmmu.py
```

4. Build merged manifests:

```bash
PYTHONPATH=src python3 scripts/data/build_sft_manifest.py --config configs/data/sft_build.yaml
PYTHONPATH=src python3 scripts/data/build_rl_manifest.py --config configs/data/rl_build.yaml
PYTHONPATH=src python3 scripts/data/build_eval_manifest.py --config configs/data/eval_build.yaml
```

5. Inspect a manifest report if needed:

```bash
PYTHONPATH=src python3 scripts/data/dataset_report.py data/manifests/sft_manifest.jsonl
```

### Manual dataset notes

Current default stance:
- `PlantVillage`, `PlantDoc`, and `IP102` have normalization adapters but are still staged through manual raw-data slots by default so the repo does not hardcode a community mirror or silently drift from the source release.
- `PlantVillageVQA`, `AgBase`, `MIRAGE`, `Agri-LLaVA`, and `AgMMU` are treated as manual slots unless the engineer places approved exports locally.
- Benchmark test data must never be copied into SFT or RL manifests.

See [`docs/data_plan.md`](docs/data_plan.md) for the full dataset plan.

## Unified schema

Normalized JSONL rows use one schema with these top-level fields:
- `sample_id`
- `source_dataset`
- `task_type`
- `split`
- `images`
- `messages`
- `target`
- `metadata`
- `verifier`
- `reward_meta`

Schema validation is enforced with Pydantic and malformed rows fail loudly.

## SFT

Default path:
- base model: `Qwen/Qwen3-VL-4B-Instruct`
- LoRA / QLoRA-style fine-tuning
- vision encoder frozen by default
- mixed-task batching across classification, VQA, consultation, and clarify/respond samples

Dry run:

```bash
PYTHONPATH=src python3 scripts/train/train_sft.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_lora.yaml \
  --dry-run
```

Real run:

```bash
PYTHONPATH=src python3 scripts/train/train_sft.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_lora.yaml
```

Optional full fine-tuning config:

```bash
PYTHONPATH=src python3 scripts/train/train_sft.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/sft_full_optional.yaml
```

## RL post-training

Default path:
- start from the SFT checkpoint, not raw base weights
- use `trl.GRPOTrainer`
- default config keeps `loss_type: grpo`
- optional `dr_grpo` and other GRPO-family losses are configurable without rewriting the codebase
- reward functions are modular and independently testable

Dry run:

```bash
PYTHONPATH=src python3 scripts/train/train_rl_grpo.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/rl_grpo_lora.yaml \
  --dry-run
```

Real run:

```bash
PYTHONPATH=src python3 scripts/train/train_rl_grpo.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --train-config configs/train/rl_grpo_lora.yaml
```

Optional larger run:

```bash
PYTHONPATH=src python3 scripts/train/train_rl_grpo.py \
  --model-config configs/model/qwen_vlm_8b.yaml \
  --train-config configs/train/rl_grpo_optional_8b.yaml
```

Reward design is documented in [`docs/reward_design.md`](docs/reward_design.md).

## Evaluation

Local holdout:

```bash
PYTHONPATH=src python3 scripts/eval/eval_local_holdout.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --eval-config configs/eval/local_holdout.yaml
```

AgMMU:

```bash
PYTHONPATH=src python3 scripts/eval/eval_agmmu.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --eval-config configs/eval/agmmu.yaml
```

MIRAGE-MMST:

```bash
PYTHONPATH=src python3 scripts/eval/eval_mirage_mmst.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --eval-config configs/eval/mirage_mmst.yaml
```

MIRAGE-MMMT:

```bash
PYTHONPATH=src python3 scripts/eval/eval_mirage_mmmt.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --eval-config configs/eval/mirage_mmmt.yaml
```

See [`docs/eval_plan.md`](docs/eval_plan.md) for evaluation details.

## Minimal smoke-test workflow

The smoke pipeline does not require public datasets or model weights. It creates tiny synthetic images and normalized manifests, then exercises the data build, SFT dry-run, RL dry-run, and oracle evaluation path.

```bash
bash scripts/run_smoke_test.sh
```

## Known limitations

- Real SFT and RL training still require the full GPU environment and the documented Python `3.10+` stack.
- The default repo does not auto-download license-sensitive or ambiguous dataset sources.
- Classification datasets only provide label-grounded supervision by default; consultation quality is strongest when MIRAGE, AgBase, or approved instruction-style agricultural data are added.
- The default SFT collator trains on the full rendered sequence rather than an assistant-only loss mask.

## High-priority next steps

- Add official-source dataset download adapters where licenses and stable URLs are confirmed.
- Add assistant-only loss masking for multimodal SFT.
- Add optional vLLM-backed RL generation when the environment is known-good.
- Add richer ontology mappings for crop, disease, pest, and management synonyms.
