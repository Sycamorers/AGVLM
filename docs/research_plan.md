# Research Plan

## Active Milestone

Run full-data SFT with `microsoft/Phi-4-reasoning-vision-15B` on the max-3-image
agricultural train/eval split. The model path remains constrained to
ground-level RGB agricultural consultation.

## Evaluation

After a completed SFT checkpoint exists, run:

- local holdout
- MIRAGE MMST
- MIRAGE MMMT

Generation metrics should run as separate jobs so long SFT jobs are not blocked
by distributed generation.
