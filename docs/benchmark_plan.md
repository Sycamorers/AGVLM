# Benchmark Plan

## Benchmark Matrix

| Benchmark | Role | Preparation | Current Integration | Paper Use |
| --- | --- | --- | --- | --- |
| MIRAGE | Primary consultation benchmark | Automatic Hugging Face download plus normalization | implemented for MMST and MMMT | main external consultation and clarify-vs-respond result |
| AgMMU | Knowledge-intensive agriculture benchmark | manual staging until access and schema are verified | planned | agricultural expert knowledge grounding |
| AgroBench | Broad agricultural capability benchmark | manual staging until official source and schema are pinned | planned | breadth and generalization |
| Local holdout | Internal deployment-relevant benchmark | generated from normalized local manifests | implemented | in-distribution and ambiguity-heavy analysis |

## Status Command

```bash
PYTHONPATH=src python scripts/benchmarks/benchmark_status.py \
  --download-mode full \
  --fraction 1.0
```

Outputs:

- `outputs/artifacts/reports/benchmark_status.json`
- `outputs/artifacts/reports/benchmark_status.md`

The status report distinguishes implemented benchmarks from planned or blocked benchmarks and lists missing required paths.

## MIRAGE

Prepare:

```bash
PYTHONPATH=src python scripts/data/download_public_datasets.py --download-mode full --fraction 1.0 --datasets mirage
PYTHONPATH=src python scripts/data/normalize_all.py --download-mode full --fraction 1.0
PYTHONPATH=src python scripts/data/build_eval_manifest.py --download-mode full --fraction 1.0
```

Evaluate:

```bash
PYTHONPATH=src python scripts/eval/run_benchmark.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --tasks mirage_mmst mirage_mmmt \
  --prediction-mode model \
  --checkpoint-path <checkpoint_or_adapter_dir> \
  --output-dir outputs/benchmarks/<model_name>
```

## Local Holdout

Prepare:

```bash
PYTHONPATH=src python scripts/data/normalize_all.py --download-mode full --fraction 1.0
PYTHONPATH=src python scripts/data/build_eval_manifest.py --download-mode full --fraction 1.0
```

Evaluate:

```bash
PYTHONPATH=src python scripts/eval/run_benchmark.py \
  --model-config configs/model/qwen_vlm_4b.yaml \
  --tasks local_holdout \
  --prediction-mode model \
  --checkpoint-path <checkpoint_or_adapter_dir> \
  --output-dir outputs/benchmarks/<model_name>
```

## AgMMU and AgroBench

These are explicitly planned, not silently integrated.

Required before use:

- verify official download source, license, and citation requirements
- stage raw data under `data/raw/agmmu/full` or `data/raw/agrobench/full`
- add normalizers under `scripts/data/` and `src/agri_vlm/data/`
- add `configs/eval/agmmu_full.yaml` or `configs/eval/agrobench_full.yaml`
- add task entries to `scripts/eval/run_benchmark.py`
- record readiness with `scripts/benchmarks/benchmark_status.py`

Until those steps are complete, the status report will mark them as planned or blocked.
