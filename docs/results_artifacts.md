# Results Artifacts

## Training Curves

Export curves and metric tables from a training run:

```bash
PYTHONPATH=src python scripts/artifacts/export_training_artifacts.py \
  --run-dir outputs/sft/qwen3-vl-4b-lora-full-l4
```

Outputs:

- `outputs/artifacts/tables/<run_name>/training_metrics.csv`
- `outputs/artifacts/figures/<run_name>/loss.png`
- `outputs/artifacts/figures/<run_name>/loss.pdf`
- `outputs/artifacts/figures/<run_name>/learning_rate.png`
- `outputs/artifacts/figures/<run_name>/grad_norm.png`
- `outputs/artifacts/reports/<run_name>/training_artifact_manifest.json`

The manifest records any missing metric groups so absent reward or clarify curves are explicit.
If the current environment lacks `matplotlib`, the script still writes the CSV and manifest, records `plotting_error`, and skips figures until dependencies are reinstalled from `pyproject.toml`.

## Benchmark Tables

Export model comparison tables from benchmark summaries:

```bash
PYTHONPATH=src python scripts/artifacts/export_benchmark_tables.py \
  --run Base-Qwen3-VL-4B outputs/benchmarks/base-qwen3-vl-4b_local_holdout_256 \
  --run Agri-SFT outputs/benchmarks/sft-qwen3-vl-4b-lora-full-l4
```

Outputs:

- `outputs/artifacts/tables/benchmark_results.csv`
- `outputs/artifacts/tables/benchmark_results.md`
- `outputs/artifacts/reports/benchmark_results_manifest.json`

## Paper Reuse Rules

- Regenerate figures from raw JSONL metrics instead of manually editing plots.
- Keep benchmark `summary.json` files with the table artifacts.
- Cite the exact checkpoint path and `run_metadata.json` for every reported row.
- Use PDFs for paper drafts and PNGs for quick inspection.
