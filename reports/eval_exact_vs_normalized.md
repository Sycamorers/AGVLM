# Exact-match vs Normalized Classification Metrics

## Classification metrics

| run | n | raw_exact_acc | answer_field_exact_acc | normalized_acc | macro_f1 | weighted_f1 | balanced_acc | ambiguous_rate | invalid_rate | label_mentioned_rate | oos_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage5 | 382 | 0.0000 | 0.0314 | 0.0314 | 0.0030 | 0.0044 | 0.0210 | 0.0000 | 0.0000 | 0.0314 | 0.0916 |
| stage6_mc | 382 | 0.0000 | 0.0236 | 0.0236 | 0.0031 | 0.0029 | 0.0245 | 0.0000 | 0.0000 | 0.0236 | 0.0000 |
| stage7_label_only_classification | 382 | 0.0288 | 0.0288 | 0.0288 | 0.0017 | 0.0023 | 0.0210 | 0.0000 | 0.0000 | 0.0288 | 0.3508 |

## Source prediction modes

| run | source_dataset | mode_prediction | mode_count | total | mode_rate |
| --- | --- | --- | --- | --- | --- |
| stage5 | banana_disease | black sigatoka | 17 | 17 | 1.0000 |
| stage5 | digigreen_crop_disease | healthy | 18 | 18 | 1.0000 |
| stage5 | ip102 | alfalfa weevil | 123 | 123 | 1.0000 |
| stage5 | plantdoc | bell pepper leaf spot | 21 | 21 | 1.0000 |
| stage5 | plantvillage | corn maize northern leaf blight | 104 | 104 | 1.0000 |
| stage5 | rice_disease | bacterial leaf blight | 82 | 82 | 1.0000 |
| stage5 | tea_sickness | gray blight | 17 | 17 | 1.0000 |
| stage6_mc | banana_disease | black sigatoka | 17 | 17 | 1.0000 |
| stage6_mc | digigreen_crop_disease | arhar aphids | 18 | 18 | 1.0000 |
| stage6_mc | ip102 | alfalfa weevil | 123 | 123 | 1.0000 |
| stage6_mc | plantdoc | bell pepper leaf spot | 21 | 21 | 1.0000 |
| stage6_mc | plantvillage | corn maize northern leaf blight | 104 | 104 | 1.0000 |
| stage6_mc | rice_disease | rice gall midge | 82 | 82 | 1.0000 |
| stage6_mc | tea_sickness | gray light | 17 | 17 | 1.0000 |
| stage7_label_only_classification | banana_disease | to spot | 17 | 17 | 1.0000 |
| stage7_label_only_classification | digigreen_crop_disease | aphids | 18 | 18 | 1.0000 |
| stage7_label_only_classification | ip102 | aphids | 123 | 123 | 1.0000 |
| stage7_label_only_classification | plantdoc | peach leaf | 21 | 21 | 1.0000 |
| stage7_label_only_classification | plantvillage | peach bacterial spot | 104 | 104 | 1.0000 |
| stage7_label_only_classification | rice_disease | to spot | 82 | 82 | 1.0000 |
| stage7_label_only_classification | tea_sickness | to spot | 17 | 17 | 1.0000 |

## Normalization examples

_No rows._

Confusion matrices are emitted as `reports/confusion_matrix_<run>.csv`.
Per-class precision/recall/F1 tables are emitted as `reports/per_class_metrics_<run>.csv`.

## Missing prediction artifacts

- `stage7_label_only_mixed=/blue/hmedeiros/qinruoyao/agvlm/benchmarks/vlm_baselines/results/agvlm_stage7_label_only_mixed_benchmark_20260607/predictions/sft-benchmark-agvlm-phi4-sft-stage7-label-only-mixed-b200-candidate-test.jsonl`
