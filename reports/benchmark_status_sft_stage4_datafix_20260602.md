# Benchmark Status Report

- phase: `sft`
- overall ok: `True`
- errors: `0`
- warnings: `3`

| Status | Severity | Check |
| --- | --- | --- |
| ok | error | sft_benchmark has no train/eval sample-id or group overlap. |
| ok | error | sft_benchmark split manifests have no duplicate sample IDs. |
| ok | error | External baseline model config parses. |
| ok | warning | AGVLM checkpoint config parses; placeholder paths are warnings until selected for a run. |
| ok | error | Prediction parser handles Answer, Decision, and structured sections. |
| ok | error | Metrics module can score synthetic benchmark predictions. |
| ok | error | Summary table can be refreshed. |
| ok | error | Required benchmark Slurm scripts exist. |
| fail | warning | Required benchmark/project docs exist. |
| fail | warning | SFT training-related files are dirty in the worktree. |

## Dirty SFT Guard

The status check reports dirty SFT training files if present, but it does not revert user work.

```text
benchmarks/vlm_baselines/agvlm_checkpoint_models.yaml
benchmarks/vlm_baselines/dataset_adapter.py
benchmarks/vlm_baselines/slurm/run_sft_benchmark_24gb.sbatch
src/agri_vlm/data/builders.py
tests/test_manifest_builders.py
benchmarks/vlm_baselines/splits_stage4_datafix/
configs/data/eval_build_stage4_datafix.yaml
configs/data/sft_eval_stage4_closed_label_datafix_phi4_max3.yaml
configs/data/sft_format_audit_stage4_closed_label_datafix_phi4_max3.yaml
configs/data/sft_stage4_closed_label_datafix_phi4_max3.yaml
configs/data/sft_train_eval_phi4_max3_stage4_datafix.yaml
configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage4_datafix.yaml
configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage4_datafix_preflight.yaml
reports/benchmark_status_sft_20260602.json
reports/benchmark_status_sft_20260602.md
reports/benchmark_status_sft_stage4_datafix_20260602.json
reports/benchmark_status_sft_stage4_datafix_20260602.md
reports/sft_stage3_benchmark_20260602/
reports/sft_stage4_datafix/
scripts/data/build_closed_label_eval_manifest.py
```
