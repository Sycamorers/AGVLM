# Stage3 SFT Benchmark Decision

- Benchmark status: `completed`
- Decision: **PASS active-baseline gate, do not auto-promote over Stage2**
- Recommendation: keep Stage2 B200 as the safer promotion candidate until classification collapse is fixed or an explicit tradeoff is accepted.

## Metrics

| Model | Task macro | Invalid | Class acc | Class macro-F1 | Class OOS | VQA relaxed | Clarify macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `active_previous_sft` | 0.228774 | 119 | 0.000000 | 0.000000 | 1.000000 | 0.156000 | 0.530323 |
| `stage2_b200_retry_full` | 0.302209 | 0 | 0.035088 | 0.001325 | 0.166667 | 0.224000 | 0.681301 |
| `stage2_b200_closed_label_prompt` | 0.301862 | 0 | 0.008772 | 0.000285 | 0.000000 | 0.224000 | 0.681301 |
| `stage3_closed_label_candidate` | 0.306393 | 0 | 0.008772 | 0.000342 | 0.000000 | 0.244000 | 0.674839 |

## Stage3 vs Stage2 Gate

| Required metric | Stage2 | Stage3 | Preferred delta | Pass |
| --- | ---: | ---: | ---: | --- |
| `task_macro_average` | 0.302209 | 0.306393 | 0.004185 | yes |
| `vqa_relaxed_accuracy` | 0.224000 | 0.244000 | 0.020000 | yes |
| `clarify_macro_f1` | 0.681301 | 0.674839 | -0.006462 | no |
| `num_invalid_predictions` | 0 | 0 | 0 | yes |

## Stage3 Diagnostics

- Adapter validation: `320` non-empty LoRA tensors in `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-closed-label-classification-repair-stage3-b200-4gpu/adapter_model.safetensors`.
- Benchmark: `392` samples on `cuda:0`, quantization `4bit`, OOM fallback `False`.
- Training: `1200` steps, train loss `3.255404`, final eval loss `3.969760`.
- Eval loss rose from `3.665608` at step 200 to `3.969760` at step 1200.
- Classification collapse: all `114` classification outputs parsed as exact, but only two normalized predictions appeared: `{'rice gall midge': 95, 'rice stemfly': 19}`.
- Clarify/respond parse statuses: `{'exact': 12, 'inferred': 16}`.

## Artifacts

- Stage3 metrics: `benchmarks/vlm_baselines/results/agvlm_stage3_closed_label_classification_repair_benchmark_20260601/metrics/sft-benchmark_agvlm-phi4-sft-closed-label-classification-repair-stage3-b200-candidate_test_metrics.json`
- Stage3 predictions: `benchmarks/vlm_baselines/results/agvlm_stage3_closed_label_classification_repair_benchmark_20260601/predictions/sft-benchmark-agvlm-phi4-sft-closed-label-classification-repair-stage3-b200-candidate-test.jsonl`
- Stage3 adapter validation: `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-closed-label-classification-repair-stage3-b200-4gpu/adapter_validation.json`
- Benchmark status report: `reports/benchmark_status_sft_20260602.json`
