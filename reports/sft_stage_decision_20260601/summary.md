# AGVLM SFT Stage Decision

- Benchmark status: `completed`
- Decision: **PROMOTE**
- Stage2 task macro: `0.265`
- Active SFT task macro: `0.229`

## Promotion Gate

| Metric | Baseline | Candidate | Preferred Delta | Pass |
| --- | ---: | ---: | ---: | --- |
| task_macro_average | 0.229 | 0.265 | 0.036 | yes |
| short_vqa.relaxed_accuracy | 0.156 | 0.224 | 0.068 | yes |
| clarify_or_respond.macro_f1 | 0.530 | 0.568 | 0.038 | yes |
| num_invalid_predictions | 119 | 83.000 | 36.000 | yes |

## Output Format Diagnostics

### Classification-Repair Pilot

- Invalid: `185` / `392` (`47.2%`)
- Format-like invalid: `185` (`47.2%`)
- Out-of-label parseable answers: `0` (`0.0%`)
- Explicit clarify `Decision:` rate: `57.1%`

### Stage2 B200 Candidate

- Invalid: `83` / `392` (`21.2%`)
- Format-like invalid: `83` (`21.2%`)
- Out-of-label parseable answers: `19` (`4.8%`)
- Explicit clarify `Decision:` rate: `75.0%`
