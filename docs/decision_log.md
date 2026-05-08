# Decision Log

## 2026-05-08

- Switched the active SFT path to `microsoft/Phi-4-reasoning-vision-15B`.
- Removed stale prior-model configs, scheduler wrappers, generated manifests,
  local cache entries, and Orange checkpoint/run directories.
- Kept the V1 scope to ground-level RGB agricultural consultation tasks.
- Added a Phi-4 reasoning vision collator path because the model expects
  `<image>` placeholders and flat image batches.
- Added a 16-GPU Turin Slurm wrapper that tests per-device batch-size candidates
  before launching the full run.
