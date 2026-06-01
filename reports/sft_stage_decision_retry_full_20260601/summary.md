# AGVLM SFT Stage Decision

- Benchmark status: `completed`
- Decision: **PROMOTE**
- Stage2 task macro: `0.302`
- Active SFT task macro: `0.229`

## Promotion Gate

| Metric | Baseline | Candidate | Preferred Delta | Pass |
| --- | ---: | ---: | ---: | --- |
| task_macro_average | 0.229 | 0.302 | 0.073 | yes |
| short_vqa.relaxed_accuracy | 0.156 | 0.224 | 0.068 | yes |
| clarify_or_respond.macro_f1 | 0.530 | 0.681 | 0.151 | yes |
| num_invalid_predictions | 119 | 0.000 | 119 | yes |

## Output Format Diagnostics

### Classification-Repair Pilot

- Invalid: `185` / `392` (`47.2%`)
- Format-like invalid: `185` (`47.2%`)
- Format contract issues: `257` (`65.6%`)
- Out-of-label parseable answers: `0` (`0.0%`)
- Explicit clarify `Decision:` rate: `57.1%`

### Stage2 B200 Candidate

- Invalid: `0` / `392` (`0.0%`)
- Format-like invalid: `0` (`0.0%`)
- Format contract issues: `67` (`17.1%`)
- Out-of-label parseable answers: `19` (`4.8%`)
- Explicit clarify `Decision:` rate: `75.0%`
