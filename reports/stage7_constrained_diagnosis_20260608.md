# Stage7 Constrained Classification Diagnosis

Date: 2026-06-08

## Run

- Job: `34120560`
- State: completed successfully, exit `0:0`
- Runtime: `00:16:31`
- Command path: `scripts/hpc/run_stage7_constrained_classification_benchmark.slurm`
- Output directory: `benchmarks/vlm_baselines/results/agvlm_stage7_label_only_classification_constrained_benchmark_20260608_113812`
- Prediction file: `benchmarks/vlm_baselines/results/agvlm_stage7_label_only_classification_constrained_benchmark_20260608_113812/predictions/sft-benchmark-agvlm-phi4-sft-stage7-label-only-classification-b200-candidate-test.jsonl`
- Metrics file: `benchmarks/vlm_baselines/results/agvlm_stage7_label_only_classification_constrained_benchmark_20260608_113812/metrics/sft-benchmark_agvlm-phi4-sft-stage7-label-only-classification-b200-candidate_test_metrics.json`

The run used the Stage7 label-only classification LoRA with constrained closed-label decoding for classification rows only. It did not retrain.

## Result

| mode | cls top1 | cls macro F1 | cls weighted F1 | balanced acc | OOS rate | task macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage7 free generation | 2.88% | 0.17% | 0.23% | 2.10% | 35.08% | 29.56% |
| Stage7 constrained labels | 4.97% | 0.60% | 0.60% | 4.55% | 0.00% | 28.17% |

Constrained decoding removed out-of-label-space outputs, but it did not fix classification. The model still predicts one label per source.

## Source Collapse

| source | n | constrained mode | mode rate | accuracy |
| --- | ---: | --- | ---: | ---: |
| ip102 | 123 | aphids | 100.00% | 4.88% |
| plantvillage | 104 | peach bacterial spot | 100.00% | 3.85% |
| rice_disease | 82 | rice gall midge | 100.00% | 3.66% |
| plantdoc | 21 | peach leaf | 100.00% | 4.76% |
| digigreen_crop_disease | 18 | potato healthy | 100.00% | 5.56% |
| banana_disease | 17 | black sigatoka | 100.00% | 11.76% |
| tea_sickness | 17 | algal leaf | 100.00% | 11.76% |

## Decision

The poor classification result is not mainly an output parser or exact-match metric issue. It is also not solved by forcing valid labels at decode time. Stage7 has learned source-level label priors rather than image-discriminative classification.

Do not launch another broad mixed-source SFT round. The next useful diagnosis is a micro-overfit/single-source verification:

1. Train on one source only, preferably `rice_disease` or `banana_disease`, with 5-10 classes and a tiny balanced train split.
2. Evaluate both train-set and held-out accuracy with constrained labels.
3. If train-set accuracy cannot approach 95-100%, debug image loading, target masking, adapter attachment, and Phi-4 vision gradient flow.
4. If train-set accuracy is high but held-out remains poor, the issue is data quality/generalization; use source-specific adapters and more data per class.

The current evidence argues for specialized classification adapters or a classifier-style candidate scoring head/path, not one general LoRA adapter for all classification and generation tasks.
