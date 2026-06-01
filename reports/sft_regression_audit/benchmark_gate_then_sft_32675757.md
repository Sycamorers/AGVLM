# Benchmark Gate Then SFT Submit

- Benchmark job: `32675757`
- Decision: **reject**
- Metrics path: `benchmarks/vlm_baselines/results/agvlm_previous_sft_benchmark_watch_dtypefix_v2/attempt-2/metrics/sft-benchmark_agvlm-phi4-sft-completed_test_metrics.json`
- Training job: ``

| Gate | Actual | Requirement | Pass |
| --- | ---: | ---: | --- |
| num_examples | 392 | >= 392 | yes |
| failure_rate | 1.0 | <= 0.0 | no |
| invalid_prediction_rate | 1.0 | <= 0.5 | no |
| task_macro_average | None | >= 0.13 | no |
| vqa.relaxed_accuracy | None | >= 0.12 | no |
| clarify_or_respond.macro_f1 | None | >= 0.2 | no |

Reason: Benchmark metrics did not pass configured gates.
