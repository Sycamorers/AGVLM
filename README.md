# Agri-VLM

This repository is a config-driven research codebase for an agriculture-focused
vision-language model. V1 is scoped to ground-level RGB agricultural
consultation: crop disease, pest, symptom, management, and clarify-vs-respond
tasks.

## Active Training Path

As of May 8, 2026, the active SFT model is
`microsoft/Phi-4-reasoning-vision-15B` on the full max-3-image agricultural SFT
split using 16 Turin L4 GPUs and about 800 GB aggregate node RAM.

Active files:

- data split config: `configs/data/sft_train_eval_phi4_max3.yaml`
- model config: `configs/model/phi4_reasoning_vision_15b_turin_24g.yaml`
- full train config: `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_full_max3.yaml`
- preflight train config: `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_full_max3_preflight.yaml`
- Slurm wrapper: `scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm`

Generated max3 manifests:

- `data/manifests/full/sft_train_phi4_max3_no_eval_overlap.jsonl`
- `data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl`
- `data/manifests/full/sft_train_eval_phi4_max3_summary.json`

The Slurm wrapper runs a short batch-size preflight over
`PHI4_BATCH_CANDIDATES` before the full run. The default candidate list is `1`,
which keeps the per-rank image fan-in bounded while gradient accumulation
preserves the intended effective global batch. Preflight diagnostics write under
`outputs/preflight/`; full training artifacts must write under `outputs/sft/`.

## Launch

```bash
sbatch scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm
```

Useful overrides:

```bash
sbatch \
  --export=ALL,PHI4_BATCH_CANDIDATES="2 1" \
  scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm
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

The active split is built from decode/aspect-valid source manifests and removes
train/eval image-group overlap. Full public datasets and manual datasets remain
gated by their existing documented staging steps; smoke tests must not download
full datasets or model weights.

## Tests

```bash
pytest
```

For a config-only SFT dry run:

```bash
PYTHONPATH=src python scripts/train/train_sft.py \
  --model-config configs/model/phi4_reasoning_vision_15b_turin_24g.yaml \
  --train-config configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_full_max3_preflight.yaml \
  --dry-run
```
