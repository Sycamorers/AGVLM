# RL Evaluation Report

- Manifest: `data/manifests/full/rl_local_holdout_eval.jsonl`
- Prediction mode: `model`
- Checkpoint path: `/orange/hmedeiros/qinruoyao/agvlm/outputs/rl/test/grpo-phi4-turin-16gpu-step-eval-4bit-from-sft1700`
- Examples: `2.0`

## Overall

| Metric | Value |
| --- | ---: |
| num_examples | 2.0000 |
| classification_label_accuracy | 0.0000 |
| accepted_answer_accuracy | 0.0000 |
| synonym_soft_score | 0.0000 |
| structured_section_compliance | 0.4000 |
| clarify_decision_accuracy | 0.0000 |
| management_keyword_coverage | 0.0000 |
| hallucination_forbidden_claim_rate | 0.0000 |
| average_completion_length | 33.5000 |
| average_composite_reward | 0.2000 |

## Examples

- `bad` `vqa` score=`0.0000` sample=`plantvillage_vqa-image_014634.JPG-184553`
- `borderline` `consultation` score=`0.4000` sample=`agbase-agbase-838846`
- `good` `consultation` score=`0.4000` sample=`agbase-agbase-838846`
