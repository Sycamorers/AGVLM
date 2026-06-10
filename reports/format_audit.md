# Input/Output Format Audit

## Findings

- Classification manifest targets are stored as bare canonical labels, but instructional SFT renders them as `Answer: <label>` plus `Evidence: ...`.
- Stage6 multiple-choice classification renders `Choice: <letter>`, `Answer: <label>`, and `Evidence: ...`, so it is a different output contract from Stage5 closed-label classification.
- Benchmark classification prompts ask for `Answer` plus optional evidence. Raw-output exact string matching against bare labels is therefore too strict for any output that follows the training format.
- Existing benchmark scoring already uses parser-based normalized labels; the audit adds raw-output exact, extracted-answer exact, normalized-label comparison, and examples where parsing changes the answer.
- The parser now supports line-start Markdown answer fields and JSON `answer`/`label` fields, and still marks multi-label mentions as ambiguous.

## Classification format by task/source

| task_name | task_type | output_format | risk |
| --- | --- | --- | --- |
| banana_disease:classification | classification | bare_canonical_label_manifest=465 | none_observed |
| digigreen_crop_disease:classification | classification | bare_canonical_label_manifest=5778 | large_label_space; eval_labels_missing_from_train=4 |
| ip102:classification | classification | bare_canonical_label_manifest=13331 | large_label_space |
| plantdoc:classification | classification | bare_canonical_label_manifest=14357 | large_label_space |
| plantvillage:classification | classification | bare_canonical_label_manifest=19560 | large_label_space |
| rice_disease:classification | classification | bare_canonical_label_manifest=8265 | large_label_space |
| tea_sickness:classification | classification | bare_canonical_label_manifest=546 | none_observed |

## Strict vs normalized impact

| run | raw_output_exact_accuracy | answer_field_exact_accuracy | normalized_label_accuracy | normalization_changed_rate | label_mentioned_rate |
| --- | --- | --- | --- | --- | --- |
| stage5 | 0.00% | 3.14% | 3.14% | 0.00% | 3.14% |
| stage6_mc | 0.00% | 2.36% | 2.36% | 0.00% | 2.36% |
| stage7_label_only_classification | 2.88% | 2.88% | 2.88% | 0.00% | 2.88% |

## Recommendation

For classification, standardize the target contract to label-only for classification-specific adapters or keep `Answer: <label>` but evaluate by an explicit parser. Do not mix `Answer/Evidence` and `Choice/Answer/Evidence` within the same classification benchmark without tracking them as separate tasks.
