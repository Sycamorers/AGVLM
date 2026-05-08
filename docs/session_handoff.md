# Session Handoff

## Current Goal

Run Phi-4 reasoning vision full-data SFT on 16 Turin L4 GPUs.

## Submit

```bash
sbatch scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm
```

The wrapper rebuilds the max3 manifests, verifies access to
`microsoft/Phi-4-reasoning-vision-15B`, tests per-device batch-size candidates,
and launches the full run with the first passing candidate. Preflight
diagnostics write under `outputs/preflight/`, while full training artifacts
write under `outputs/sft/` and `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/`.

## Key Files

- `configs/data/sft_train_eval_phi4_max3.yaml`
- `configs/model/phi4_reasoning_vision_15b_turin_24g.yaml`
- `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_full_max3.yaml`
- `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_full_max3_preflight.yaml`
- `scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm`
