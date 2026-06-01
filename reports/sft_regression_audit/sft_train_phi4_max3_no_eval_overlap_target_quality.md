# SFT Target Quality Audit

- Manifest: `data/manifests/full/sft_train_phi4_max3_no_eval_overlap.jsonl`
- Rows: `292514`
- Target format: `instructional`

## Task Mix

| Task | Rows |
| --- | ---: |
| clarify_or_respond | 6482 |
| classification | 86228 |
| consultation | 60114 |
| vqa | 139690 |

## Target Lengths

| Field | Count | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| rendered_target_token_lengths | 292514 | 2 | 5.0 | 28.599 | 1155 |
| answer_token_lengths | 292514 | 1 | 4.0 | 27.826 | 1155 |

## Flags

| Flag | Count | Rate |
| --- | ---: | ---: |
| classification_numeric_label_prefix | 40624 | 0.138879 |
| short_answer | 94230 | 0.322138 |

## Top Rendered Answers

| Answer | Count |
| --- | ---: |
| no | 35095 |
| yes | 23881 |
| tomato | 6571 |
| orange haunglongbing citrus greening | 4406 |
| tomato tomato yellow leaf curl virus | 4286 |
| soybean healthy | 4072 |
| 102 cicadellidae | 3073 |
| diagnosis: huanglongbing (hlb), or citrus greening | 3023 |
| the leaf shows blotchy, asymmetrical yellowing, a classic symptom of hlb | 2963 |
| the diagnosis is tomato yellow leaf curl virus (tylcv) | 2962 |
| the causal agent is the tomato yellow leaf curl virus | 2958 |
| the cause is the bacterium candidatus liberibacter asiaticus, leading to huanglongbing | 2955 |
| this is citrus greening disease | 2888 |
| 68 lycorma delicatula | 2859 |
| this is a viral infection: tylcv | 2846 |
| this tomato leaf shows classic signs of tylcv, like yellowing and curling | 2841 |
| 71 miridae | 2780 |
| 25 aphids | 2217 |
| orange | 1966 |
| peach bacterial spot | 1837 |
| tomato bacterial spot | 1712 |
| the large, dark, water-soaked lesions are a key sign of late blight | 1643 |
| the plant is suffering from a late blight infection | 1585 |
| this is late blight, caused by the oomycete phytophthora infestans | 1567 |
| tomato septoria leaf spot | 1548 |
| the condition is identified as late blight | 1544 |
| tomato late blight | 1527 |
| squash powdery mildew | 1468 |
| the numerous small, angular spots are characteristic of bacterial spot | 1416 |
| this is bacterial spot. note the small, dark, water-soaked lesions | 1414 |

## Examples: `short_answer`

| # | Dataset | Task | Sample ID | Target |
| ---: | --- | --- | --- | --- |
| 1 | plantvillage | classification | `plantvillage-train-001228` | Answer: apple healthy |
| 2 | plantvillage | classification | `plantvillage-train-001229` | Answer: apple healthy |
| 3 | plantvillage | classification | `plantvillage-train-001230` | Answer: apple healthy |
| 4 | plantvillage | classification | `plantvillage-train-001231` | Answer: apple healthy |
| 5 | plantvillage | classification | `plantvillage-train-001232` | Answer: apple healthy |
| 6 | plantvillage | classification | `plantvillage-train-001233` | Answer: apple healthy |
| 7 | plantvillage | classification | `plantvillage-train-001234` | Answer: apple healthy |
| 8 | plantvillage | classification | `plantvillage-train-001235` | Answer: apple healthy |
| 9 | plantvillage | classification | `plantvillage-train-001236` | Answer: apple healthy |
| 10 | plantvillage | classification | `plantvillage-train-001237` | Answer: apple healthy |

## Examples: `classification_numeric_label_prefix`

| # | Dataset | Task | Sample ID | Target |
| ---: | --- | --- | --- | --- |
| 1 | ip102 | classification | `ip102-ip102-v1-1-images-00002-jpg` | Answer: 1 rice leaf roller |
| 2 | ip102 | classification | `ip102-ip102-v1-1-images-00003-jpg` | Answer: 1 rice leaf roller |
| 3 | ip102 | classification | `ip102-ip102-v1-1-images-00005-jpg` | Answer: 1 rice leaf roller |
| 4 | ip102 | classification | `ip102-ip102-v1-1-images-00008-jpg` | Answer: 1 rice leaf roller |
| 5 | ip102 | classification | `ip102-ip102-v1-1-images-00011-jpg` | Answer: 1 rice leaf roller |
| 6 | ip102 | classification | `ip102-ip102-v1-1-images-00015-jpg` | Answer: 1 rice leaf roller |
| 7 | ip102 | classification | `ip102-ip102-v1-1-images-00017-jpg` | Answer: 1 rice leaf roller |
| 8 | ip102 | classification | `ip102-ip102-v1-1-images-00018-jpg` | Answer: 1 rice leaf roller |
| 9 | ip102 | classification | `ip102-ip102-v1-1-images-00019-jpg` | Answer: 1 rice leaf roller |
| 10 | ip102 | classification | `ip102-ip102-v1-1-images-00021-jpg` | Answer: 1 rice leaf roller |
