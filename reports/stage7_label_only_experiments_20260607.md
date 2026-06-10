# Stage7 Label-only Classification Experiments

Date: 2026-06-07

## Baseline already completed

Experiment A was completed by the audit reports:

- Stage5 raw-output exact accuracy: 0.0000
- Stage5 extracted-answer / normalized classification accuracy: 0.0314
- Stage5 classification macro F1: 0.0030
- Stage6 MC normalized classification accuracy: 0.0236
- Stage6 MC classification macro F1: 0.0031

Conclusion: raw exact matching was too strict, but parser normalization did not rescue classification. The main failure is source-level label collapse.

## Implemented format path

Added `classification_label_only` SFT format:

- Classification prompt: asks for exactly one allowed label and only the selected label text.
- Classification target: bare canonical label.
- Non-classification rows: keep existing instructional target contracts.

Added label-only benchmark prompt support through:

```bash
AGRI_VLM_CLASSIFICATION_PROMPT_FORMAT=label_only
```

## Experiment B: Format-standardized mixed SFT

Purpose: keep Stage5 mixed data content, but train classification rows with bare-label outputs.

Train config:

```text
configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage7_label_only_mixed.yaml
```

Preflight config:

```text
configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage7_label_only_mixed_preflight.yaml
```

Output adapter:

```text
/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-stage7-label-only-mixed-b200-4gpu
```

Original Slurm job:

```text
34070652
```

Status: canceled before start to avoid parallel large-memory group usage.

Dependent benchmark job:

```text
34070661
```

Status: canceled before start. Submit only after the active sequential run finishes.

Benchmark output directory:

```text
benchmarks/vlm_baselines/results/agvlm_stage7_label_only_mixed_benchmark_20260607
```

## Experiment C: Classification-only label-only SFT

Purpose: train only classification data with a strict label-only contract.

Data config:

```text
configs/data/sft_classification_only_stage7_label_only_phi4_max3.yaml
```

Train manifest:

```text
data/manifests/full/sft_classification_only_stage7_label_only_train.jsonl
```

Train rows: 61,632 classification rows.

Train config:

```text
configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage7_label_only_classification.yaml
```

Preflight config:

```text
configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage7_label_only_classification_preflight.yaml
```

Output adapter:

```text
/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-stage7-label-only-classification-b200-4gpu
```

Original Slurm job:

```text
34070653
```

Status: canceled before start to reduce queue footprint.

Sequential replacement Slurm job:

```text
34071393
```

Request:

```text
4 GPUs, 32 CPUs, 256 GB memory
```

Dependent benchmark job:

```text
34070662
```

Status: canceled before start. Submit after job `34071393` completes successfully.

Benchmark output directory:

```text
benchmarks/vlm_baselines/results/agvlm_stage7_label_only_classification_benchmark_20260607
```

## Shared validation manifest

Cleaned validation manifest:

```text
data/manifests/full/sft_classification_stage7_label_only_val.jsonl
```

Rows: 288 classification rows from the frozen benchmark validation split.

Coverage note: validation contains IP102, rice, and tea. Final comparison must use the frozen test benchmark because test covers all classification sources.

## Monitor commands

```bash
squeue -j 34070652,34070653,34070661,34070662 -o '%i %j %T %M %D %R'
sacct -j 34070652,34070653,34070661,34070662 --format=JobID,JobName,State,ExitCode,Elapsed -P
```

## Next after B/C finish

1. Let only job `34071393` run.
2. Validate its saved adapter with `adapter_validation.json`.
3. Submit the classification-only benchmark only after `34071393` completes successfully.
4. Re-run `scripts/analysis/audit_sft_scope.py` after the benchmark finishes.
5. Decide whether to submit Experiment B, also as a single sequential job.

Experiment D should not synthesize data blindly. It should start only after B/C determine whether label-only formatting and task separation reduce collapse. If collapse persists, add curated or source-approved examples for underperforming classes and use balanced sampling.

Experiment E is the comparison across Stage5, B, and C using the same frozen split and the same parser. It becomes actionable when jobs `34070661` and `34070662` complete.

## Dependent report refresh

The original audit/report refresh was submitted as a CPU dependency on both benchmark jobs:

```text
34070758
```

Status: canceled before start. It should be resubmitted only after the sequential benchmark completes.

It reruns:

```bash
PYTHONPATH=src:benchmarks/vlm_baselines python scripts/analysis/audit_sft_scope.py --overwrite
```

This will update `reports/eval_exact_vs_normalized.md`, confusion matrices, per-class metrics, and error analysis with Stage7 B/C prediction artifacts if both dependent benchmarks complete successfully.
