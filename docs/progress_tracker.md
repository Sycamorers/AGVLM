# Progress Tracker

Current active milestone: Phi-4 reasoning vision full-data SFT on the max-3-image manifest using 16 Turin L4 GPUs.

| Area | Status | Notes |
| --- | --- | --- |
| SFT split | ready | `sft_train_phi4_max3_no_eval_overlap.jsonl` and `sft_eval_phi4_max3_stratified512.jsonl` exist. |
| Active SFT run | active/pending external to this task | Do not interrupt server training. |
| Benchmark split builder | ready | `build_phase_splits.py` writes phase-labeled SFT and RL val/test manifests. |
| SFT benchmark harness | ready for dry-run | External baselines and completed SFT checkpoints are supported. Full benchmark not run. |
| RL benchmark harness | ready for dry-run | External baselines, completed SFT, and completed RL checkpoints are supported. Full benchmark not run. |
| Metrics | ready | Task-specific metrics for classification, short VQA, clarify, and consultation are implemented. |
| Checkpoint config | placeholder | `agvlm_checkpoint_models.yaml` must be updated with real SFT/RL paths before project checkpoint runs. |
| RL training | blocked | Must wait for completed SFT checkpoint. RL must not start from the raw base model. |
| Reports | ready | Benchmark audit and split/status reports are under `reports/` and `benchmarks/vlm_baselines/splits/`. |

## Current Split Counts

- SFT benchmark: 120 val rows, 392 test rows, zero exact/group overlap with SFT train.
- RL benchmark: 369 val rows, 1,573 test rows after filtering 2,154 RL local-holdout rows with train image-group overlap.

## Immediate Next Actions

1. Let SFT finish.
2. Replace `agvlm_phi4_sft_completed` placeholder paths with the completed SFT checkpoint or adapter.
3. Run the SFT checkpoint dry-run command from `docs/benchmark_plan.md`.
4. Run intended SFT benchmark jobs.
5. Start RL only from the completed SFT checkpoint after SFT benchmark review.
