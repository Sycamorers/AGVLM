# Project Plan

## Scope

V1 targets an agriculture-specialized VLM for ground-level RGB consultation only.

Included:
- plant disease identification
- insect and pest identification
- symptom explanation
- structured agricultural consultation
- clarify-vs-respond behavior

Explicitly not prioritized in V1:
- aerial imagery
- remote sensing
- 3D reasoning
- generic all-purpose multimodal chat

## Training stages

Stage A:
- prepare raw dataset slots
- normalize each source into one JSONL schema
- build SFT, RL, and evaluation manifests

Stage B:
- run SFT from `Qwen/Qwen3-VL-4B-Instruct`
- default to LoRA with the vision tower frozen
- keep `Qwen/Qwen3-VL-8B-Instruct` as an optional larger config

Stage C:
- run GRPO post-training on top of the SFT checkpoint
- restrict RL data to rewardable or semi-verifiable tasks
- keep GRPO-family loss changes config-switchable

## Engineering principles

- config-driven execution
- no silent dataset skipping
- fail loudly on malformed normalized rows
- smoke-testable without public datasets
- maintainable wrappers over mainstream libraries
