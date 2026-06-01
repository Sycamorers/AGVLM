# RL Reward Sanity Check

- Manifest: `data/manifests/full/rl_manifest.jsonl`
- Sampled rows: `200`
- Reward modules: `exact_match, normalized_label, synonym_match, structured_format, uncertainty_calibration, clarify_vs_respond, management_coverage, hallucination_penalty, output_format`
- Assertion failure count: `0`

## Average Reward By Candidate

| Candidate | Average Reward |
| --- | ---: |
| empty | -1.0000 |
| generic_clarify | -0.0600 |
| generic_overconfident | -2.8494 |
| generic_uncertain | -0.0362 |
| known_bad | -0.2856 |
| known_good | 1.5455 |
| structured_consultation | 0.5443 |
| target_answer | 1.5455 |

## Average Reward By Task Type

| Task Type | Average Reward |
| --- | ---: |
| clarify_or_respond | -0.5938 |
| classification | 0.2124 |
| consultation | -1.1481 |
| vqa | -0.1444 |

## Average Reward By Verifier Mode

| Verifier Mode | Average Reward |
| --- | ---: |
| clarify | -0.5938 |
| exact_match | -0.1444 |
| label | 0.2124 |
| structured | -1.1481 |

## Reward Distribution

| Statistic | Reward |
| --- | ---: |
| min | -4.6500 |
| p25 | -1.0000 |
| median | 0.0000 |
| p75 | 1.0000 |
| p95 | 2.5000 |
| max | 2.5000 |

## Examples

## Assertion Failures

No assertion failures.
