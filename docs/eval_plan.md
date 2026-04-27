# Eval Plan

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
- clarify precision and recall when decision labels are present
- unnecessary clarification rate
- premature answer rate
- average composite reward for held-out outputs

## Smoke evaluation

The smoke pipeline uses:
- synthetic normalized data
- dry-run SFT and RL
- oracle local evaluation

This verifies repository wiring without requiring benchmark-only datasets or model downloads.
