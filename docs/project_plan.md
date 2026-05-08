# Project Plan

## V1 Scope

Build and evaluate an agriculture-focused vision-language model for
ground-level RGB consultation. Avoid making generic multimodal assistant
behavior the default training path.

## Active Work

- run SFT from `microsoft/Phi-4-reasoning-vision-15B`
- use the max-3-image non-overlapping agricultural split
- keep long-run generation evaluation separate from training
- evaluate the completed SFT checkpoint before GRPO
