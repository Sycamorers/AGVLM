# SFT Promotion Gate

- Candidate: `agvlm_phi4_sft_balanced_v2_instructional_completed`
- Baseline: `agvlm_phi4_sft_completed`
- Decision: **REJECT**

| Required Metric | Baseline | Candidate | Preferred Delta | Required Margin | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| task_macro_average | 0.228774 | 0.207030 | -0.021744 | 0.000000 | no |
| short_vqa.relaxed_accuracy | 0.156000 | 0.212000 | 0.056000 | 0.000000 | yes |
| clarify_or_respond.macro_f1 | 0.530323 | 0.409091 | -0.121232 | 0.000000 | no |
| num_invalid_predictions | 119 | 193 | -74.000000 | 0 | no |

| Diagnostic Metric | Baseline | Candidate | Preferred Delta |
| --- | ---: | ---: | ---: |
| classification.top1_accuracy | 0.000000 | 0.000000 | 0.000000 |
| classification.macro_f1 | 0.000000 | 0.000000 | 0.000000 |
| local_metrics.average_reward |  |  |  |

Failed required metrics:
- `task_macro_average`
- `clarify_or_respond.macro_f1`
- `num_invalid_predictions`
