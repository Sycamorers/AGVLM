# Results Artifacts

## Current Artifact State

Local `outputs/` and `logs/` were cleaned on May 6, 2026 after the failed AGBASE-disjoint continuation was diagnosed. The retained SFT artifact on Orange is:

```text
/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft
```

Future probe and full max3 runs should write fresh local run directories under `outputs/sft/` and checkpoint copies under `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/`.

## Training Curves

Export curves and metric tables from a training run:

```bash
PYTHONPATH=src python scripts/artifacts/export_training_artifacts.py \
  --run-dir outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced
```

Probe example:

```bash
PYTHONPATH=src python scripts/artifacts/export_training_artifacts.py \
  --run-dir outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced-probe
```

Outputs:

- `outputs/artifacts/tables/<run_name>/training_metrics.csv`
- `outputs/artifacts/figures/<run_name>/loss.png`
- `outputs/artifacts/figures/<run_name>/loss.pdf`
- `outputs/artifacts/figures/<run_name>/learning_rate.png`
- `outputs/artifacts/figures/<run_name>/grad_norm.png`
- `outputs/artifacts/reports/<run_name>/training_artifact_manifest.json`

The manifest records any missing metric groups so absent reward or clarify curves are explicit. If the current environment lacks `matplotlib`, the script still writes the CSV and manifest, records `plotting_error`, and skips figures until dependencies are reinstalled from `pyproject.toml`.

## Benchmark Tables

Export model comparison tables from benchmark summaries:

```bash
PYTHONPATH=src python scripts/artifacts/export_benchmark_tables.py \
  --run Llama4-Balanced outputs/benchmarks/llama4-balanced \
  --run Llama4-Full-Max3 outputs/benchmarks/llama4-scout-full-max3
```

Use real benchmark output directories for reported rows. Do not use a raw checkpoint directory as a benchmark result unless it contains benchmark `summary.json` files.

Outputs:

- `outputs/artifacts/tables/benchmark_results.csv`
- `outputs/artifacts/tables/benchmark_results.md`
- `outputs/artifacts/reports/benchmark_results_manifest.json`

## Paper Reuse Rules

- Regenerate figures from raw JSONL metrics instead of manually editing plots.
- Keep benchmark `summary.json` files with the table artifacts.
- Cite the exact checkpoint path and `run_metadata.json` for every reported row.
- Mark the failed AGBASE-disjoint continuation as excluded from final model selection unless it is later rerun with corrected data/eval settings.
- Use PDFs for paper drafts and PNGs for quick inspection.
