# Experiment Roadmap

## Current Run

1. Build Phi-4 max3 train/eval manifests.
2. Verify model access and runtime packages.
3. Test per-device batch-size candidates on 16 Turin L4 GPUs.
4. Launch full SFT with the largest passing candidate.
5. Export training artifacts after metrics are available.

## Next Runs

After SFT completes:

- run local holdout and MIRAGE benchmarks
- run separate generation evaluation
- decide whether the checkpoint is strong enough to seed GRPO
