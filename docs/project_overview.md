# Project Overview

## Scope

This repository is the operating system for a ground-level RGB agricultural VLM paper. V1 is scoped to agricultural consultation from ordinary field or crop images, with emphasis on disease identification, pest identification, symptom interpretation, management-oriented advice, and clarify-vs-respond behavior.

The default path is not a generic all-purpose VLM assistant. All data preparation, training, evaluation, and paper artifacts should support agricultural consultation.

## Paper Thesis

General-purpose VLMs often answer too early when agricultural evidence is incomplete. This project formulates agricultural consultation as a decision-aware task: the model should either answer when the image and context are sufficient, or ask a high-value clarification question when they are not.

Primary contribution candidates:

- Clarify-vs-respond task definition for agricultural consultation.
- Decision-aware data construction with direct-answer, clarify-first, and clarify-to-resolution examples.
- Two-stage post-training: agricultural SFT followed by policy optimization for consultation decision behavior.
- Reliability-oriented evaluation beyond answer accuracy.

## Repository Roles

- Training codebase: SFT and GRPO entrypoints under `scripts/train/`, with library code under `src/agri_vlm/training/`.
- Benchmark codebase: local holdout and MIRAGE entrypoints today, with registry support for AgMMU and AgroBench readiness tracking.
- Artifact system: reusable metrics, figures, tables, reports, resolved configs, and run metadata under `outputs/`.
- Progress reference: project stage, blockers, active milestone, and next actions tracked in `docs/progress_tracker.md`.

## Output Convention

```text
outputs/
  sft/<run_name>/
    resolved_config.yaml
    run_metadata.json
    artifact_manifest.json
    metrics/train_metrics.jsonl
    metrics.jsonl
    tensorboard/
    artifacts/
    checkpoint-*/
  rl/<run_name>/
    resolved_config.yaml
    run_metadata.json
    artifact_manifest.json
    metrics/train_metrics.jsonl
    tensorboard/
  benchmarks/<model_or_run_name>/
    summary.json
    <task>/metrics.json
    <task>/predictions.jsonl
  artifacts/
    figures/
    tables/
    reports/
```

Rank-zero training writes the provenance files and structured metric logs. TensorBoard event files are written by the trainer integration.
