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

## Problem framing by stage

Stage A solves the data consistency problem. The selected agriculture datasets use different
formats, labels, splits, image layouts, licenses, and access rules. The pipeline converts them
into one auditable schema and separates SFT, RL, and evaluation data so training can be repeated
without manual path edits or silent dataset omissions.

Stage B solves the domain adaptation problem. The base VLM has general multimodal ability, but it
does not know the desired agriculture-specific label space, consultation style, or output behavior.
SFT teaches the model to connect crop imagery, disease and pest labels, VQA answers, and
management-style responses.

Stage C solves the behavior alignment problem. SFT can make the model imitate examples, but it does
not directly optimize for conservative agricultural consultation. GRPO is used to reward verifiable
answers, clarify-vs-respond decisions, uncertainty when evidence is incomplete, structured outputs,
management coverage, and penalties for unsupported claims.

Evaluation solves the measurement problem. The local holdout and MIRAGE tasks are intended to show
whether fine-tuning improves agriculture-specific recognition and consultation behavior under the
same conditions used for the base model.

## Engineering principles

- config-driven execution
- no silent dataset skipping
- fail loudly on malformed normalized rows
- smoke-testable without public datasets
- maintainable wrappers over mainstream libraries
