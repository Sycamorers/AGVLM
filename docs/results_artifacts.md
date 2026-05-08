# Results Artifacts

## Active SFT Output

The Phi-4 Turin wrapper writes the full run under a batch-specific directory:

```text
outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch<N>
/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch<N>
```

Use the selected batch directory from the Slurm log for benchmark exports and
paper tables. Do not use a raw checkpoint directory as a benchmark result unless
it contains benchmark `summary.json` files.
