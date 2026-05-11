# Session Handoff

## Current State

SFT remains the active training stage. The current intended SFT path is `microsoft/Phi-4-reasoning-vision-15B` on the full max-3-image agricultural SFT split. Do not interrupt a running SFT job.

The benchmark framework is prepared for two phases:

- `sft_benchmark`: external baselines plus completed SFT checkpoint.
- `rl_benchmark`: external baselines plus completed SFT and completed RL checkpoints.

No full benchmark run has been executed as part of this preparation.

## Key Benchmark Files

- `benchmarks/vlm_baselines/build_phase_splits.py`
- `benchmarks/vlm_baselines/prediction_parsing.py`
- `benchmarks/vlm_baselines/metrics.py`
- `benchmarks/vlm_baselines/run_baselines.py`
- `benchmarks/vlm_baselines/evaluate_predictions.py`
- `benchmarks/vlm_baselines/agvlm_checkpoint_models.yaml`
- `scripts/benchmarks/benchmark_status.py`

## Before SFT Benchmark

Update `benchmarks/vlm_baselines/agvlm_checkpoint_models.yaml`:

- set `agvlm_phi4_sft_completed.checkpoint_path` for a merged checkpoint, or `adapter_path` for LoRA
- remove placeholder values from unused path fields
- keep `base_model_name_or_path: microsoft/Phi-4-reasoning-vision-15B`

Then run:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split val \
  --model-key agvlm_phi4_sft_completed \
  --max-samples 2 \
  --dry-run
```

## Before RL

RL must start from the completed SFT checkpoint or adapter. Do not start RL from the raw base model.

Before RL benchmark, update `agvlm_phi4_rl_completed` with:

- real RL checkpoint or adapter path
- `initialized_from_sft_checkpoint` pointing to the SFT checkpoint used to initialize RL
- no placeholder checkpoint path fields

Then run:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase rl \
  --split val \
  --model-key agvlm_phi4_rl_completed \
  --max-samples 2 \
  --dry-run
```

## Known Limitation

The RL benchmark split is leakage-filtered. The source RL local holdout has 4,096 rows, but 2,154 rows share image groups with the RL train manifest and are excluded from benchmark val/test outputs. Report this in final results.
