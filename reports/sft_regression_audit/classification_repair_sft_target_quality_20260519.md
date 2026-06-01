# SFT Target Quality Audit

- Manifest: `data/manifests/full/sft_train_phi4_max3_classification_repair_instructional.jsonl`
- Rows: `167710`
- Target format: `instructional`

## Task Mix

| Task | Rows |
| --- | ---: |
| clarify_or_respond | 6482 |
| classification | 86228 |
| consultation | 25000 |
| vqa | 50000 |

## Target Lengths

| Field | Count | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| rendered_target_token_lengths | 167710 | 2 | 14.0 | 25.490 | 902 |
| answer_token_lengths | 167710 | 1 | 13.0 | 24.678 | 902 |

## Flags

| Flag | Count | Rate |
| --- | ---: | ---: |
| classification_numeric_label_prefix | 40624 | 0.242228 |
| short_answer | 25714 | 0.153324 |

## Top Rendered Answers

| Answer | Count |
| --- | ---: |
| no | 12489 |
| yes | 8504 |
| orange haunglongbing citrus greening evidence: visible orange symptoms support the haunglongbing (citrus greening) label | 4406 |
| tomato tomato yellow leaf curl virus evidence: visible tomato symptoms support the tomato yellow leaf curl virus label | 4286 |
| soybean healthy evidence: visible soybean symptoms support the healthy label | 4072 |
| 102 cicadellidae evidence: visible agricultural symptoms or pest features support this label | 3073 |
| 68 lycorma delicatula evidence: visible agricultural symptoms or pest features support this label | 2859 |
| 71 miridae evidence: visible agricultural symptoms or pest features support this label | 2780 |
| tomato | 2386 |
| 25 aphids evidence: visible agricultural symptoms or pest features support this label | 2217 |
| peach bacterial spot evidence: visible peach symptoms support the bacterial spot label | 1837 |
| tomato bacterial spot evidence: visible tomato symptoms support the bacterial spot label | 1712 |
| tomato late blight evidence: visible tomato symptoms support the late blight label | 1527 |
| squash powdery mildew evidence: visible squash symptoms support the powdery mildew label | 1468 |
| tomato septoria leaf spot evidence: visible tomato symptoms support the septoria leaf spot label | 1417 |
| tomato spider mites two-spotted spider mite evidence: visible tomato symptoms support the spider mites two-spotted spider mite label | 1341 |
| apple healthy evidence: visible apple symptoms support the healthy label | 1336 |
| tomato healthy evidence: visible tomato symptoms support the healthy label | 1307 |
| blueberry healthy evidence: visible blueberry symptoms support the healthy label | 1198 |
| pepper bell healthy evidence: visible pepper, bell symptoms support the healthy label | 1182 |
| diagnosis: huanglongbing (hlb), or citrus greening | 1125 |
| tomato target spot evidence: visible tomato symptoms support the target spot label | 1123 |
| grape esca black measles evidence: visible grape symptoms support the esca (black measles) label | 1106 |
| the causal agent is the tomato yellow leaf curl virus | 1088 |
| the cause is the bacterium candidatus liberibacter asiaticus, leading to huanglongbing | 1079 |
| the leaf shows blotchy, asymmetrical yellowing, a classic symptom of hlb | 1056 |
| 52 blister beetle evidence: visible agricultural symptoms or pest features support this label | 1036 |
| this is a viral infection: tylcv | 1032 |
| the diagnosis is tomato yellow leaf curl virus (tylcv) | 1028 |
| this is citrus greening disease | 1027 |

## Examples: `classification_numeric_label_prefix`

| # | Dataset | Task | Sample ID | Target |
| ---: | --- | --- | --- | --- |
| 1 | ip102 | classification | `ip102-ip102-v1-1-images-58881-jpg` | Answer: 75 Panonchus citri McGregor<br>Evidence: Visible agricultural symptoms or pest features support this label. |
| 2 | ip102 | classification | `ip102-ip102-v1-1-images-27121-jpg` | Answer: 39 cabbage army worm<br>Evidence: Visible agricultural symptoms or pest features support this label. |
| 3 | ip102 | classification | `ip102-ip102-v1-1-images-00861-jpg` | Answer: 1 rice leaf roller<br>Evidence: Visible agricultural symptoms or pest features support this label. |
| 4 | ip102 | classification | `ip102-ip102-v1-1-images-52254-jpg` | Answer: 70 Cicadella viridis<br>Evidence: Visible agricultural symptoms or pest features support this label. |
| 5 | ip102 | classification | `ip102-ip102-v1-1-images-32605-jpg` | Answer: 48 tarnished plant bug<br>Evidence: Visible agricultural symptoms or pest features support this label. |

## Examples: `short_answer`

| # | Dataset | Task | Sample ID | Target |
| ---: | --- | --- | --- | --- |
| 1 | plantvillage_vqa | vqa | `plantvillage_vqa-image_006760.JPG-019768` | Answer: Tomato |
| 2 | plantvillage_vqa | vqa | `plantvillage_vqa-image_008056.JPG-023603` | Answer: No |
| 3 | plantvillage_vqa | vqa | `plantvillage_vqa-image_034704.JPG-188956` | Answer: Yes |
| 4 | plantvillage_vqa | vqa | `plantvillage_vqa-image_029294.JPG-086209` | Answer: Yes |
| 5 | plantvillage_vqa | vqa | `plantvillage_vqa-image_013150.JPG-038482` | Answer: Yes |
