# AGENTS.md

This repository is a config-driven research codebase for an agriculture-focused vision-language model.

Working rules:
- Keep V1 scoped to ground-level RGB agricultural consultation only.
- Do not add generic all-purpose VLM behavior as the default path.
- Preserve explicit documentation for any gated, licensed, or manual dataset steps.
- Prefer config changes over hardcoded paths or one-off script edits.
- Keep all code comments in English.
- Do not silently skip datasets, splits, or reward modules when inputs are missing.

Implementation conventions:
- Python package code lives under `src/agri_vlm/`.
- Data normalization scripts live under `scripts/data/`.
- Training scripts are thin wrappers around library code under `src/agri_vlm/training/`.
- Evaluation scripts are thin wrappers around library code under `src/agri_vlm/evaluation/`.
- Smoke tests must work without downloading full public datasets or model weights.
