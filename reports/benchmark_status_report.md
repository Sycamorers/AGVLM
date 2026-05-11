# Benchmark Status Report

- phase: `both`
- overall ok: `True`
- errors: `0`
- warnings: `2`

| Status | Severity | Check |
| --- | --- | --- |
| ok | error | rl_benchmark has no train/eval sample-id or group overlap. |
| ok | error | rl_benchmark split manifests have no duplicate sample IDs. |
| ok | error | sft_benchmark has no train/eval sample-id or group overlap. |
| ok | error | sft_benchmark split manifests have no duplicate sample IDs. |
| ok | error | External baseline model config parses. |
| ok | warning | AGVLM checkpoint config parses; placeholder paths are warnings until selected for a run. |
| ok | error | Prediction parser handles Answer, Decision, and structured sections. |
| ok | error | Metrics module can score synthetic benchmark predictions. |
| ok | error | Summary table can be refreshed. |
| ok | error | Required benchmark Slurm scripts exist. |
| ok | info | Required benchmark/project docs exist. |
| fail | warning | SFT training-related files are dirty in the worktree. |

## Dirty SFT Guard

The status check reports dirty SFT training files if present, but it does not revert user work.

```text
Makefile
benchmarks/vlm_baselines/README.md
benchmarks/vlm_baselines/agvlm_checkpoint_models.yaml
benchmarks/vlm_baselines/build_phase_splits.py
benchmarks/vlm_baselines/checkpoint_config.py
benchmarks/vlm_baselines/dataset_adapter.py
benchmarks/vlm_baselines/evaluate_predictions.py
benchmarks/vlm_baselines/metrics.py
benchmarks/vlm_baselines/model_adapters.py
benchmarks/vlm_baselines/prediction_parsing.py
benchmarks/vlm_baselines/run_baselines.py
benchmarks/vlm_baselines/scripts/run_rl_benchmark_all_baselines.sh
benchmarks/vlm_baselines/scripts/run_rl_benchmark_one_model.sh
benchmarks/vlm_baselines/scripts/run_rl_benchmark_rl_checkpoint.sh
benchmarks/vlm_baselines/scripts/run_rl_benchmark_sft_checkpoint.sh
benchmarks/vlm_baselines/scripts/run_sft_benchmark_agvlm_checkpoint.sh
benchmarks/vlm_baselines/scripts/run_sft_benchmark_all_baselines.sh
benchmarks/vlm_baselines/scripts/run_sft_benchmark_one_model.sh
benchmarks/vlm_baselines/slurm/run_rl_benchmark_24gb.sbatch
benchmarks/vlm_baselines/slurm/run_rl_benchmark_agvlm_checkpoint.sbatch
benchmarks/vlm_baselines/slurm/run_sft_benchmark_24gb.sbatch
benchmarks/vlm_baselines/slurm/run_sft_benchmark_agvlm_checkpoint.sbatch
benchmarks/vlm_baselines/split_dataset.py
benchmarks/vlm_baselines/splits/benchmark_split_report.json
benchmarks/vlm_baselines/splits/benchmark_split_report.md
benchmarks/vlm_baselines/splits/distribution_report.json
benchmarks/vlm_baselines/splits/rl_test_manifest.jsonl
benchmarks/vlm_baselines/splits/rl_val_manifest.jsonl
benchmarks/vlm_baselines/splits/sft_test_manifest.jsonl
benchmarks/vlm_baselines/splits/sft_val_manifest.jsonl
benchmarks/vlm_baselines/splits/test_manifest.jsonl
benchmarks/vlm_baselines/splits/val_manifest.jsonl
configs/data/rl_build.yaml
configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_full_max3.yaml
docs/benchmark_plan.md
docs/eval_plan.md
docs/experiment_roadmap.md
docs/progress_tracker.md
docs/project_overview.md
docs/project_plan.md
docs/results_artifacts.md
docs/rlft_design.md
docs/rlft_pipeline.md
docs/session_handoff.md
docs/training_monitoring.md
reports/benchmark_framework_audit.json
reports/benchmark_framework_audit.md
reports/benchmark_status_report.json
reports/benchmark_status_report.md
scripts/artifacts/export_benchmark_tables.py
scripts/benchmarks/benchmark_status.py
scripts/data/audit_rl_manifest.py
scripts/data/build_rl_manifest.py
scripts/hpc/run_rl_grpo_b200_4gpu_phi4_reasoning_vision_15b.slurm
scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm
scripts/train/check_rl_dataset_format.py
scripts/train/rl_reward_sanity_check.py
scripts/train/train_rl_grpo.py
src/agri_vlm/data/builders.py
src/agri_vlm/evaluation/inference.py
src/agri_vlm/rewards/clarify_decision.py
src/agri_vlm/rewards/classification.py
src/agri_vlm/rewards/composite.py
src/agri_vlm/rewards/exact_match.py
src/agri_vlm/rewards/hallucination_penalty.py
src/agri_vlm/rewards/management_coverage.py
src/agri_vlm/rewards/structure.py
src/agri_vlm/rewards/synonym_match.py
src/agri_vlm/rewards/uncertainty.py
src/agri_vlm/schemas/config_schema.py
src/agri_vlm/schemas/reward_schema.py
src/agri_vlm/training/rl_trainer.py
src/agri_vlm/utils/checkpointing.py
tests/test_benchmark_checkpoint_config.py
tests/test_benchmark_metrics.py
tests/test_benchmark_phase_splits.py
tests/test_benchmark_prediction_parsing.py
tests/test_benchmark_summary_table.py
tests/test_reward_functions.py
tests/test_rl_readiness_pipeline.py
configs/model/phi4_reasoning_vision_15b_b200.yaml
configs/model/phi4_reasoning_vision_15b_turin_24g_4bit_rl.yaml
configs/train/rl_grpo_phi4_full.yaml
configs/train/rl_grpo_phi4_readiness.yaml
configs/train/rl_grpo_phi4_smoke.yaml
configs/train/rl_grpo_phi4_turin_16gpu_step_eval_4bit_from_sft1700.yaml
configs/train/rl_grpo_phi4_turin_16gpu_step_eval_from_sft1700.yaml
configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_full_max3.yaml
configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_full_max3_preflight.yaml
reports/rl_data_prep_report.json
reports/rl_data_prep_report.md
reports/rl_eval_metrics.json
reports/rl_eval_report.md
reports/rl_eval_samples.jsonl
reports/rl_prep_report.md
reports/rl_turin_step_eval_metrics.json
reports/rl_turin_step_eval_report.md
reports/rl_turin_step_eval_samples.jsonl
scripts/data/prepare_rl_datasets.py
scripts/eval/eval_rl_checkpoint.py
scripts/hpc/guard_cancel_turin_after_b200_ready.sh
scripts/hpc/guard_cancel_turin_after_b200_ready.slurm
scripts/hpc/run_rl_grpo_phi4_full.slurm
scripts/hpc/run_rl_grpo_phi4_smoke.slurm
scripts/hpc/run_rl_grpo_phi4_turin_16gpu_step_eval.slurm
scripts/hpc/run_sft_b200_4gpu_phi4_reasoning_vision_15b_full_max3.slurm
scripts/hpc/submit_b200_and_guard_turin.sh
src/agri_vlm/rewards/parsing.py
tests/test_checkpointing.py
```
