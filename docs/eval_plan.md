# Eval Plan

## AgMMU

Local implementation:
- exact match for MCQ-style answers
- exact match for short open-ended answers where feasible

Usage:
- evaluation-only manifest
- keep benchmark test rows out of SFT and RL manifests

## MIRAGE-MMST

Local implementation:
- answer exact match where references are short
- structured consultation reward signals for management-style outputs

## MIRAGE-MMMT

Local implementation:
- clarify/respond decision accuracy
- answer exact match where references are deterministic

## Local holdout

Built from non-overlapping PlantDoc, IP102, and PlantVillageVQA samples.

Reported metrics:
- label accuracy
- macro F1 for label tasks
- exact match for short-answer tasks
- clarify accuracy when applicable
- average composite reward for held-out outputs

## Smoke evaluation

The smoke pipeline uses:
- synthetic normalized data
- dry-run SFT and RL
- oracle local evaluation

This verifies repository wiring without requiring public benchmarks or model downloads.
