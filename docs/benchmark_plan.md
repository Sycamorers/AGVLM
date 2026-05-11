# Benchmark Plan

## Philosophy

Use task-specific metrics, not one generic score. Keep classification, short VQA, clarify, and consultation metrics separate, and report a task macro average so majority tasks do not hide failures. Report source-dataset breakdowns to identify overfitting to lab-style datasets.

Deterministic metrics are primary. Reward scores may be added as diagnostics for RL, but they must not replace benchmark metrics. LLM-as-judge is not part of the current primary protocol.

## Splits

SFT benchmark:

- source: `data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl`
- train overlap check: `data/manifests/full/sft_train_phi4_max3_no_eval_overlap.jsonl`
- outputs:
  - `benchmarks/vlm_baselines/splits/sft_val_manifest.jsonl`
  - `benchmarks/vlm_baselines/splits/sft_test_manifest.jsonl`

RL benchmark:

- source: `data/manifests/full/rl_local_holdout_eval.jsonl`
- train overlap check: `data/manifests/full/rl_manifest.jsonl`
- outputs:
  - `benchmarks/vlm_baselines/splits/rl_val_manifest.jsonl`
  - `benchmarks/vlm_baselines/splits/rl_test_manifest.jsonl`

Current RL split note: 2,154 local-holdout rows were filtered because their image group overlapped the RL train manifest. This leaves 1,942 RL benchmark rows and should be reported in final results.

## Metrics

Classification:

- top-1 accuracy
- macro-F1
- weighted-F1
- balanced accuracy
- per-class precision/recall/F1/support
- confusion matrix
- invalid, missing, and out-of-label-space rates

Short VQA:

- exact match
- normalized exact match
- relaxed accuracy
- token-F1
- yes/no accuracy
- numeric relaxed accuracy when numeric references exist
- containment as diagnostic only

Clarify-or-respond:

- decision accuracy
- clarify/respond precision, recall, F1
- macro-F1
- confusion matrix
- over-clarification and under-clarification rates

Consultation:

- structured section compliance
- required section compliance
- management keyword coverage
- forbidden claim rate
- unsafe or overconfident claim rate
- uncertainty compliance
- follow-up question presence
- answer length and repetition diagnostics

## Commands

Build SFT benchmark splits:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/build_phase_splits.py \
  --phase sft \
  --write-report
```

Build RL benchmark splits:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/build_phase_splits.py \
  --phase rl \
  --write-report
```

Build both:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/build_phase_splits.py \
  --phase both \
  --write-report
```

Dry-run one external baseline on SFT without loading a model:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split val \
  --model-name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-samples 2 \
  --dry-run
```

Prepared, do not run until intended: one external baseline on SFT benchmark:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split test \
  --model-name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --output-dir benchmarks/vlm_baselines/results
```

Prepared, do not run until intended: all external baselines on SFT:

```bash
bash benchmarks/vlm_baselines/scripts/run_sft_benchmark_all_baselines.sh
```

Dry-run completed SFT checkpoint on SFT after updating paths:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split val \
  --model-key agvlm_phi4_sft_completed \
  --max-samples 2 \
  --dry-run
```

Prepared, do not run until intended: completed SFT checkpoint on SFT:

```bash
bash benchmarks/vlm_baselines/scripts/run_sft_benchmark_agvlm_checkpoint.sh
```

Dry-run one external baseline on RL without loading a model:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase rl \
  --split val \
  --model-name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-samples 2 \
  --dry-run
```

Prepared, do not run until intended: all external baselines on RL:

```bash
bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_all_baselines.sh
```

Dry-run completed SFT checkpoint on RL after updating paths:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase rl \
  --split val \
  --model-key agvlm_phi4_sft_completed \
  --max-samples 2 \
  --dry-run
```

Dry-run completed RL checkpoint on RL after updating paths:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase rl \
  --split val \
  --model-key agvlm_phi4_rl_completed \
  --max-samples 2 \
  --dry-run
```

Prepared, do not run until intended: completed SFT and RL checkpoints on RL:

```bash
bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_sft_checkpoint.sh
bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_rl_checkpoint.sh
```

Refresh summary tables:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/evaluate_predictions.py \
  --refresh-summary-only \
  --output-dir benchmarks/vlm_baselines/results/metrics
```

Export benchmark artifacts:

```bash
PYTHONPATH=src python3 scripts/artifacts/export_benchmark_tables.py \
  --summary-table benchmarks/vlm_baselines/results/metrics/summary_table.csv \
  --output-root outputs/artifacts
```
