# Agri-VLM

This repository is a config-driven research codebase for an agriculture-focused
vision-language model. V1 is scoped to ground-level RGB agricultural
consultation: crop disease, pest, symptom, management, and clarify-vs-respond
tasks.

## Active Training Path

As of June 3, 2026, the active SFT path is Stage5 data-fix training for
`microsoft/Phi-4-reasoning-vision-15B` on 4 B200 GPUs. Stage4 completed but was
not promoted because benchmark classification collapsed to one dominant label
per source despite lower eval loss. Stage5 therefore starts from the Stage2
adapter and uses an expanded, closed-label classification mix.

Active files:

- dataset registry: `configs/data/datasets.yaml`
- SFT source merge config: `configs/data/sft_build_stage5_datafix.yaml`
- eval/holdout config: `configs/data/eval_build_stage5_datafix.yaml`
- train/eval split config: `configs/data/sft_train_eval_phi4_max3_stage5_datafix.yaml`
- closed-label train config: `configs/data/sft_stage5_closed_label_datafix_phi4_max3.yaml`
- closed-label eval config: `configs/data/sft_eval_stage5_closed_label_datafix_phi4_max3.yaml`
- format audit config: `configs/data/sft_format_audit_stage5_closed_label_datafix_phi4_max3.yaml`
- model config: `configs/model/phi4_reasoning_vision_15b_b200.yaml`
- preflight train config: `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage5_datafix_preflight.yaml`
- full train config: `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage5_datafix.yaml`
- Slurm wrapper: `scripts/hpc/run_sft_b200_4gpu_phi4_reasoning_vision_15b_full_max3.slurm`

Generated Stage5 manifests:

- `data/manifests/full/sft_manifest_stage5_datafix.jsonl`
- `data/manifests/full/sft_manifest_stage5_datafix.decode_valid_images.jsonl`
- `data/manifests/full/local_holdout_eval_stage5_datafix.decode_valid_images.jsonl`
- `data/manifests/full/sft_train_phi4_max3_stage5_no_eval_overlap.jsonl`
- `data/manifests/full/sft_eval_phi4_max3_stage5_raw_stratified1024.jsonl`
- `data/manifests/full/sft_train_phi4_max3_stage5_closed_label_datafix.jsonl`
- `data/manifests/full/sft_eval_phi4_max3_stage5_closed_label_stratified1024.jsonl`

The Stage5 closed-label train manifest has `143,114` rows:
`61,632` classification, `50,000` VQA, `25,000` consultation, and `6,482`
clarify-or-respond. The training dry-run passed with `1,024` eval rows.
Detailed counts, validation notes, and audit outputs are in
`reports/sft_stage5_datafix/progress_20260603.md`.

## Launch

```bash
sbatch \
  --job-name=agri-vlm-sft-stage5-datafix \
  --export=ALL,DATA_CONFIG=configs/data/sft_stage5_closed_label_datafix_phi4_max3.yaml,PREFLIGHT_CONFIG=configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage5_datafix_preflight.yaml,TRAIN_CONFIG=configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage5_datafix.yaml,TENSORBOARD_PORT=6015 \
  scripts/hpc/run_sft_b200_4gpu_phi4_reasoning_vision_15b_full_max3.slurm
```

The current submitted Stage5 job is:

```text
job_id: 33840540
job_name: agri-vlm-sft-stage5-datafix
partition: hpg-b200
```

The full run writes local artifacts under `outputs/sft/` and checkpoint
artifacts under `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/`. It should not
write full-training artifacts under `outputs/smoke/`.

## RLFT Readiness

As of May 15, 2026, SFT is still the active main training stage. The current
RLFT target is rule-based / verifier-based GRPO post-training for
`microsoft/Phi-4-reasoning-vision-15B` initialized from a completed Phi-4 SFT
checkpoint or adapter. There is no pretrained learned reward model yet. The
RLFT code/data pipeline is being prepared and validated; full GRPO training has
not been launched as part of this stage, and non-dry-run GRPO must not start
from the raw base model.

The default GRPO reward remains a composite of deterministic modules: exact
match, normalized labels, synonym matching, structured format, uncertainty
calibration, clarify-vs-respond, management coverage, and hallucination
penalties. A future-compatible expert preference data path now exists, but it is
optional and does not change default rule-based GRPO behavior.

Build and validate the full reward-verifiable RL manifest:

```bash
make rl-data-full
make rl-audit-full
make rl-reward-check-full
make rl-format-check-full
make rl-phi4-readiness
```

CPU-only validation commands:

```bash
pytest tests/test_reward_functions.py tests/test_rl_manifest_validation_and_scoring.py tests/test_rl_readiness_pipeline.py

PYTHONPATH=src python3 scripts/validate_rl_manifest.py \
  --manifest data/manifests/full/rl_manifest.jsonl \
  --output-json reports/rl_manifest_validation.json

PYTHONPATH=src python3 scripts/score_rl_manifest.py \
  --manifest data/manifests/full/rl_manifest.jsonl \
  --output reports/rl_reward_report.jsonl \
  --summary-output reports/rl_reward_summary.json \
  --max-samples 200
```

Future expert preference rows can be exported for reward-model work without
training a reward model:

```bash
PYTHONPATH=src python3 scripts/data/prepare_pairwise_preference_data.py \
  --manifest data/manifests/full/rl_manifest.jsonl \
  --output data/interim/rl_pairwise_preferences.jsonl \
  --allow-empty
```

The readiness dry-run uses:

```bash
PYTHONPATH=src python scripts/train/train_rl_grpo.py \
  --model-config configs/model/phi4_reasoning_vision_15b_turin_24g.yaml \
  --train-config configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_readiness.yaml \
  --dry-run
```

After SFT completes and the checkpoint placeholder is replaced, start with the
smoke-after-SFT Slurm path:

```bash
sbatch \
  --export=ALL,TRAIN_CONFIG=configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_smoke_after_sft.yaml \
  scripts/hpc/run_rl_grpo_b200_4gpu_phi4_reasoning_vision_15b.slurm
```

For a smaller optional hpg-turin validation job, use the tiny smoke script. It
requests 8 GPUs, 96 GB total RAM, 45 minutes, 8 manifest samples, and 2 GRPO
steps, and it runs manifest validation plus reward-only scoring before model
loading:

```bash
sbatch \
  --export=ALL,SFT_CHECKPOINT_PATH=/path/to/completed/sft/checkpoint_or_adapter \
  scripts/hpc/run_rl_grpo_phi4_turin8_tiny_smoke.slurm
```

See `docs/rlft_design.md`, `docs/reward_design.md`, and
`docs/preference_reward_data.md` for reward design, preference-data format,
validation gates, commands, limitations, and the post-RL evaluation plan.

## Data

The active Stage5 split is built from decode-valid source manifests and removes
train/eval image-group overlap. Stage5 adds four public, scoped agricultural
classification datasets:

- `rice_disease`: `37,978` normalized rows
- `digigreen_crop_disease`: `1,092` normalized rows
- `banana_disease`: `777` normalized rows
- `tea_sickness`: `885` normalized rows

All four new datasets decoded cleanly. Existing `agbase` contributed `1,475`
invalid image rows during full SFT decode validation; those rows are excluded
from `sft_manifest_stage5_datafix.decode_valid_images.jsonl`.

Rice source labels `N`, `P`, and `K` are normalized to `nitrogen deficiency`,
`phosphorus deficiency`, and `potassium deficiency`. Do not train on the raw
one-letter nutrient labels.

Full public datasets and manual datasets remain gated by their documented
staging steps; smoke tests must not download full datasets or model weights.

## Tests

```bash
pytest
```

For a config-only SFT dry run:

```bash
PYTHONPATH=src python scripts/train/train_sft.py \
  --model-config configs/model/phi4_reasoning_vision_15b_b200.yaml \
  --train-config configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage5_datafix.yaml \
  --dry-run
```
