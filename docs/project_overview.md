# Project Overview

## Goal

AGVLM is a config-driven research codebase for an agriculture-specialized vision-language model. V1 targets ground-level RGB agricultural scenarios: crop disease, pest and symptom understanding, short VQA, structured consultation, management suggestions, and clarify-vs-respond behavior.

The default path is not a generic all-purpose VLM assistant. Training, data construction, evaluation, and reporting should stay scoped to agricultural consultation from field, crop, pest, disease, and plant symptom images.

## Expected Behavior

The model should:

- recognize crop, disease, pest, and visible symptom evidence
- answer short agricultural VQA questions
- provide structured agronomic reasoning when consultation is requested
- give management suggestions when appropriate
- ask clarifying questions when image/question evidence is insufficient
- avoid overconfident diagnoses and unsafe recommendations

## Stage Plan

Stage 0: data acquisition, normalization, and manifests

- Normalize public and manual agricultural datasets into repository JSONL manifests.
- Preserve explicit documentation for gated, licensed, or manual dataset steps.
- Construct train/eval splits with image-group leakage checks.

Stage 1: SFT

- Active model path: `microsoft/Phi-4-reasoning-vision-15B`.
- Active data path: full max-3-image agricultural SFT split.
- Purpose: teach task format, agricultural vocabulary, visual-language grounding, and answer style.
- Output: completed SFT checkpoint or adapter.
- Benchmark: external baselines plus completed SFT model on the SFT held-out benchmark split.

Stage 2: RL / GRPO

- Starts only from the completed SFT checkpoint or adapter.
- Purpose: improve reward-verifiable behavior: output format, accepted-answer correctness, consultation sections, clarify-vs-respond, management coverage, uncertainty control, and forbidden claim avoidance.
- Output: completed RL checkpoint or adapter.
- Benchmark: external baselines plus completed SFT and RL models on the RL held-out benchmark split.

Stage 3: result packaging

- Export benchmark tables and training curves.
- Write final reports and paper-ready tables.
- Document limitations and failure modes.

## Repository Roles

- `src/agri_vlm/`: package code.
- `scripts/data/`: data normalization and manifest construction.
- `scripts/train/`: thin training wrappers around library code.
- `scripts/eval/`: thin evaluation wrappers around library code.
- `benchmarks/vlm_baselines/`: inference-only external baseline and project checkpoint benchmark harness.
- `docs/`: project, benchmark, evaluation, and handoff documentation.
- `reports/`: readiness, audit, and stage reports.

## Output Convention

Benchmark outputs belong under:

```text
benchmarks/vlm_baselines/results/
outputs/benchmark/
reports/
docs/
```

Training outputs remain under `outputs/sft/` or `outputs/rl/`. Benchmark code must not write into training output directories.
