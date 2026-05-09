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

The current RLFT target is rule-based GRPO post-training for
`microsoft/Phi-4-reasoning-vision-15B` initialized from a completed Phi-4 SFT
checkpoint or adapter. Formal RLFT is pending SFT completion; the RLFT code/data
pipeline is prepared, but non-dry-run GRPO must not start from the raw base
model.

Build and validate the full reward-verifiable RL manifest:

```bash
make rl-data-full
make rl-audit-full
make rl-reward-check-full
make rl-format-check-full
make rl-phi4-readiness
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

See `docs/rlft_design.md` and `docs/rlft_pipeline.md` for the reward design,
gates, commands, and post-RL evaluation plan.

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
