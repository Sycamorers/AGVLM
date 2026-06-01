# SFT Promotion Gate

- Candidate: `new_sft`
- Baseline: `previous_sft`
- Decision: **REJECT**

| Required Metric | Baseline | Candidate | Preferred Delta | Required Margin | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| task_macro_average | 0.133731 | 0.066908 | -0.066823 | 0.000000 | no |
| short_vqa.relaxed_accuracy | 0.128000 | 0.020000 | -0.108000 | 0.000000 | no |
| clarify_or_respond.macro_f1 | 0.243243 | 0.166667 | -0.076577 | 0.000000 | no |
| num_invalid_predictions | 216 | 244 | -28.000000 | 0 | no |

| Diagnostic Metric | Baseline | Candidate | Preferred Delta |
| --- | ---: | ---: | ---: |
| classification.top1_accuracy | 0.029915 | 0.008547 | -0.021368 |
| classification.macro_f1 | 0.029948 | 0.014056 | -0.015892 |
| local_metrics.average_reward | -0.030526 | 0.017952 | 0.048478 |

Failed required metrics:
- `task_macro_average`
- `short_vqa.relaxed_accuracy`
- `clarify_or_respond.macro_f1`
- `num_invalid_predictions`
