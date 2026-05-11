# Eval Plan

## Evaluation Surfaces

Two evaluation surfaces are maintained:

- `sft_benchmark`: held-out SFT benchmark for external baselines and the completed SFT model.
- `rl_benchmark`: reward-verifiable held-out RL benchmark for external baselines, completed SFT model, and completed RL model.

The surfaces must remain clearly labeled in predictions, metrics, summary tables, and final reports.

## Prediction Format

Classification and short VQA should parse `Answer:` first.

Clarify tasks should parse:

```text
Decision: <clarify or respond>
Answer: <short answer or clarifying question>
```

Consultation tasks should parse line-start headers:

```text
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

Loose substrings do not count as structured section compliance.

## SFT Evaluation

Models:

- external baselines from `baseline_models.yaml`
- completed SFT checkpoint or adapter from `agvlm_checkpoint_models.yaml`

Data:

- `benchmarks/vlm_baselines/splits/sft_val_manifest.jsonl`
- `benchmarks/vlm_baselines/splits/sft_test_manifest.jsonl`

Primary decision:

- compare the completed SFT model against external baselines on agricultural classification and VQA while tracking invalid outputs and multi-image limitations.

## RL Evaluation

Models:

- external baselines
- completed SFT checkpoint
- completed RL checkpoint

Data:

- `benchmarks/vlm_baselines/splits/rl_val_manifest.jsonl`
- `benchmarks/vlm_baselines/splits/rl_test_manifest.jsonl`

Primary decision:

- compare RL against SFT to test whether reward-verifiable behavior improved without degrading core classification/VQA.

Report:

- classification and VQA metrics
- clarify decision metrics
- structured consultation diagnostics
- forbidden claim and overconfidence diagnostics
- optional reward scores only as diagnostics

## Smoke and Dry-Run Evaluation

Dry-runs validate configs and samples without model loads:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split val \
  --model-name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-samples 2 \
  --dry-run
```

CPU-safe tests cover parser, metrics, splits, checkpoint config validation, and summary table generation.
