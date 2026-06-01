# Benchmark Gate Then SFT Submit

- Benchmark job: `32676558`
- Decision: **submit_training**
- Metrics path: `benchmarks/vlm_baselines/results/agvlm_previous_sft_benchmark_watch_autocast_min2/attempt-1/metrics/sft-benchmark_agvlm-phi4-sft-completed_test_metrics.json`
- Training job: `32677813`

| Gate | Actual | Requirement | Pass |
| --- | ---: | ---: | --- |
| num_examples | 392 | >= 392 | yes |
| failure_rate | 0.0 | <= 0.0 | yes |
| invalid_prediction_rate | 0.30357142857142855 | <= 0.5 | yes |
| task_macro_average | 0.22877419354838713 | >= 0.13 | yes |
| vqa.relaxed_accuracy | 0.156 | >= 0.12 | yes |
| clarify_or_respond.macro_f1 | 0.5303225806451614 | >= 0.2 | yes |
