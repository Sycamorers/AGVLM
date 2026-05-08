# Training Monitoring

## Phi-4 SFT

Submit:

```bash
sbatch scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm
```

Check queue state:

```bash
squeue -u "$USER"
```

Check logs:

```bash
ls -lh logs/slurm/
tail -n 80 logs/slurm/agri-vlm-sft-phi4rv-16g-<jobid>.out
tail -n 80 logs/slurm/agri-vlm-sft-phi4rv-16g-<jobid>.err
```

The wrapper runs per-device batch-size preflights from
`PHI4_BATCH_CANDIDATES`, then starts the full run with the first passing batch
size. Preflight outputs are diagnostics only and live under
`outputs/preflight/`.

Full run artifacts are batch-specific:

```bash
ls -lh outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch*/
ls -lh /orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch*/
```

Preflight diagnostics are batch-specific:

```bash
ls -lh outputs/preflight/sft-phi4-reasoning-vision-15b-max3-turin-16gpu-batch*/
```
