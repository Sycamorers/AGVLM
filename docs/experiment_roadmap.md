# Experiment Roadmap

## Current Run

1. Build or verify Phi-4 max-3 train/eval manifests.
2. Run full SFT from `microsoft/Phi-4-reasoning-vision-15B`.
3. Export SFT training artifacts after the run completes.
4. Benchmark external baselines and the completed SFT checkpoint on the SFT benchmark split.
5. Decide whether the completed SFT checkpoint is strong enough to seed RL.

## SFT Benchmark Gate

Required before full SFT benchmark:

- `benchmarks/vlm_baselines/splits/sft_val_manifest.jsonl`
- `benchmarks/vlm_baselines/splits/sft_test_manifest.jsonl`
- completed SFT checkpoint or adapter path in `agvlm_checkpoint_models.yaml`
- successful dry-run with `--model-key agvlm_phi4_sft_completed`

Prepared but do not run until intended:

```bash
bash benchmarks/vlm_baselines/scripts/run_sft_benchmark_all_baselines.sh
bash benchmarks/vlm_baselines/scripts/run_sft_benchmark_agvlm_checkpoint.sh
```

## RLFT Phase Gates

RL/GRPO is prepared as rule-based post-training, not full RLHF. It must start from a completed SFT checkpoint or adapter.

Required before RL training:

- full reward-verifiable RL manifest
- RL manifest audit
- reward sanity check
- dataset format check
- completed SFT checkpoint validation
- readiness dry-run

Required before RL benchmark:

- `benchmarks/vlm_baselines/splits/rl_val_manifest.jsonl`
- `benchmarks/vlm_baselines/splits/rl_test_manifest.jsonl`
- completed SFT checkpoint entry
- completed RL checkpoint entry with `initialized_from_sft_checkpoint`
- successful SFT and RL checkpoint dry-runs on the RL benchmark split

Prepared but do not run until intended:

```bash
bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_all_baselines.sh
bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_sft_checkpoint.sh
bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_rl_checkpoint.sh
```

## Final Packaging

After full benchmark runs:

- refresh `benchmarks/vlm_baselines/results/metrics/summary_table.csv`
- export benchmark artifacts with `scripts/artifacts/export_benchmark_tables.py`
- export training artifacts with `scripts/artifacts/export_training_artifacts.py`
- write final limitations around deterministic consultation metrics and RL holdout filtering
