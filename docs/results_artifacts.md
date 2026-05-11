# Results Artifacts

## Training Artifacts

Active SFT training writes under `outputs/sft/` and mirrored HPC locations configured by the SFT Slurm wrapper. Benchmark preparation does not write to those directories.

RL training, when started later, writes under `outputs/rl/`.

## Benchmark Artifacts

Phase split manifests:

```text
benchmarks/vlm_baselines/splits/sft_val_manifest.jsonl
benchmarks/vlm_baselines/splits/sft_test_manifest.jsonl
benchmarks/vlm_baselines/splits/rl_val_manifest.jsonl
benchmarks/vlm_baselines/splits/rl_test_manifest.jsonl
benchmarks/vlm_baselines/splits/benchmark_split_report.json
benchmarks/vlm_baselines/splits/benchmark_split_report.md
```

Benchmark predictions:

```text
benchmarks/vlm_baselines/results/predictions/*.jsonl
```

Benchmark metrics:

```text
benchmarks/vlm_baselines/results/metrics/*_metrics.json
benchmarks/vlm_baselines/results/metrics/summary_table.csv
benchmarks/vlm_baselines/results/metrics/summary_table.json
benchmarks/vlm_baselines/results/metrics/summary_table.md
```

Run metadata:

```text
benchmarks/vlm_baselines/results/metadata/*_run.json
```

Reports:

```text
reports/benchmark_framework_audit.md
reports/benchmark_framework_audit.json
reports/benchmark_status_report.md
reports/benchmark_status_report.json
```

## Required Prediction Metadata

Each prediction row should include phase, split, model name/key, checkpoint type, base model, adapter/checkpoint path, sample id, source dataset, task type, image paths/count, image policy, prompt, raw output, parsed prediction, normalized prediction, parse status, invalid flag, ground truth, references, verifier mode, generation config, dtype, quantization, runtime, and error message.

## Export Commands

Refresh summary:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/evaluate_predictions.py \
  --refresh-summary-only \
  --output-dir benchmarks/vlm_baselines/results/metrics
```

Export benchmark table:

```bash
PYTHONPATH=src python3 scripts/artifacts/export_benchmark_tables.py \
  --summary-table benchmarks/vlm_baselines/results/metrics/summary_table.csv \
  --output-root outputs/artifacts
```

Export training curves after a run:

```bash
PYTHONPATH=src python3 scripts/artifacts/export_training_artifacts.py \
  --run-dir <completed_training_run_dir>
```
