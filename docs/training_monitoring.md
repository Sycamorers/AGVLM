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

## Phi-4 SFT on B200

Launch the B200 replacement job and the safety guard from the repository root:

```bash
bash scripts/hpc/submit_b200_and_guard_turin.sh
```

The helper selects the single running Turin Phi-4 job when it can identify one
unambiguously. To pin the current Turin job explicitly:

```bash
TURIN_JOB_ID=<turin_job_id> bash scripts/hpc/submit_b200_and_guard_turin.sh
```

The B200 Slurm wrapper requests one `hpg-b200` node with four B200 GPUs, runs
the max-3-image preflight config first, and starts full training only if
preflight succeeds. Full B200 outputs are:

```bash
outputs/sft/phi4-reasoning-vision-15b-full-max3-b200-4gpu/
/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-b200-4gpu/
```

Safety rule: the Turin job is cancelled only by
`scripts/hpc/guard_cancel_turin_after_b200_ready.sh`, and only after the B200
job is `RUNNING` and the full B200 run has valid real JSONL training metrics
under the full `outputs/sft/...b200-4gpu/` directory. Preflight metrics never
count.

Monitor:

```bash
squeue -j <b200_job_id>
squeue -j <turin_job_id>
squeue -j <guard_job_id>
tail -f logs/slurm/agri-vlm-sft-phi4rv-b200x4-<b200_job_id>.out
tail -f logs/slurm/agri-vlm-guard-b200-ready-<guard_job_id>.out
```

If the B200 job stays pending, the guard stays pending and Turin continues. If
B200 preflight fails, full training never starts and Turin continues. If B200
starts but full-training metrics are missing, invalid, stale relative to the
B200 start time, or only from preflight, the guard leaves Turin running.
