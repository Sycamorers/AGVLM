# Training Plan

## Active SFT

- base model: `microsoft/Phi-4-reasoning-vision-15B`
- hardware: 16 Turin L4 GPUs, about 800 GB aggregate node RAM
- precision: bf16
- data: max-3-image agricultural SFT split
- initialization: base model, no previous SFT adapter
- inline generation metrics: disabled

The Slurm wrapper performs a 2-step preflight over per-device batch-size
candidates and launches the full run with the largest passing candidate.
Preflights skip final model saving and write diagnostics under
`outputs/preflight/`; full runs write artifacts under `outputs/sft/`.

```bash
sbatch scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm
```
