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
`PHI4_BATCH_CANDIDATES` before the full run. The default candidate list is
`4 3 2 1`, and the first passing per-device batch size is used for full
training. Preflight diagnostics write under `outputs/preflight/`; full training
artifacts must write under `outputs/sft/`.

## Launch

```bash
sbatch scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm
```

Useful overrides:

```bash
sbatch \
  --export=ALL,PHI4_BATCH_CANDIDATES="3 2 1" \
  scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm
```

The full run writes local artifacts under `outputs/sft/` and checkpoint
artifacts under `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/`. It should not
write full-training artifacts under `outputs/smoke/`.

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
