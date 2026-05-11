# Inference-Only VLM Benchmarks

This directory contains the inference-only benchmark harness for AGVLM. It reads normalized manifests, runs models in `model.eval()` / `torch.no_grad()` mode, and writes benchmark artifacts under `benchmarks/vlm_baselines/results/`. It does not call training scripts or write training checkpoints.

## Phases

SFT benchmark:

- external baselines plus completed SFT checkpoint
- manifests: `splits/sft_val_manifest.jsonl`, `splits/sft_test_manifest.jsonl`
- source: `data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl`

RL benchmark:

- external baselines plus completed SFT and completed RL checkpoints
- manifests: `splits/rl_val_manifest.jsonl`, `splits/rl_test_manifest.jsonl`
- source: `data/manifests/full/rl_local_holdout_eval.jsonl`

The RL benchmark builder filters rows whose sample id or image group overlaps the RL train manifest. Current prepared RL splits have 369 val rows and 1,573 test rows after filtering.

## Models

External baselines are configured in `baseline_models.yaml`:

- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- `google/paligemma2-3b-mix-448`
- `microsoft/Phi-4-multimodal-instruct`
- `allenai/Molmo2-4B`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- `Qwen/Qwen2.5-VL-3B-Instruct`

Project checkpoints are configured in `agvlm_checkpoint_models.yaml`:

- `agvlm_phi4_sft_completed`
- `agvlm_phi4_rl_completed`

Placeholder checkpoint paths are warnings in readiness checks and fatal when selected for a run.

## Metrics

Classification:

- top-1 accuracy, macro-F1, weighted-F1, balanced accuracy
- per-class precision/recall/F1/support and confusion matrix
- invalid, missing, and out-of-label-space rates

Short VQA:

- exact match, normalized exact match, relaxed accuracy, token-F1
- yes/no accuracy and numeric relaxed accuracy where applicable

Clarify-or-respond:

- decision accuracy, clarify/respond PRF, macro-F1, confusion matrix
- over-clarification and under-clarification rates

Consultation:

- structured section compliance, required section compliance
- management keyword coverage
- forbidden claim and overconfidence rates
- uncertainty and follow-up diagnostics

## Safe Commands

Build both phase splits:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/build_phase_splits.py \
  --phase both \
  --write-report
```

Dry-run one SFT benchmark baseline without model load:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split val \
  --model-name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-samples 2 \
  --dry-run
```

Dry-run one RL benchmark baseline without model load:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase rl \
  --split val \
  --model-name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-samples 2 \
  --dry-run
```

Refresh summary table:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/evaluate_predictions.py \
  --refresh-summary-only \
  --output-dir benchmarks/vlm_baselines/results/metrics
```

Run readiness checks:

```bash
PYTHONPATH=src:benchmarks/vlm_baselines python3 scripts/benchmarks/benchmark_status.py \
  --phase both \
  --write-report
```

## Prepared Full Benchmark Commands

Do not run these until intended checkpoints and GPU budget are ready.

```bash
bash benchmarks/vlm_baselines/scripts/run_sft_benchmark_all_baselines.sh
bash benchmarks/vlm_baselines/scripts/run_sft_benchmark_agvlm_checkpoint.sh
bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_all_baselines.sh
bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_sft_checkpoint.sh
bash benchmarks/vlm_baselines/scripts/run_rl_benchmark_rl_checkpoint.sh
```

Slurm wrappers are under `slurm/` and write to `benchmarks/vlm_baselines/results/`.
