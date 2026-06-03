# SFT Target Quality Audit

- Manifest: `data/manifests/full/sft_train_phi4_max3_stage5_closed_label_datafix.jsonl`
- Rows: `143114`
- Target format: `instructional`

## Task Mix

| Task | Rows |
| --- | ---: |
| clarify_or_respond | 6482 |
| classification | 61632 |
| consultation | 25000 |
| vqa | 50000 |

## Target Lengths

| Field | Count | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| rendered_target_token_lengths | 143114 | 2 | 13.0 | 27.261 | 1155 |
| answer_token_lengths | 143114 | 1 | 12.0 | 26.481 | 1155 |

## Flags

| Flag | Count | Rate |
| --- | ---: | ---: |
| short_answer | 25812 | 0.180360 |

## Top Rendered Answers

| Answer | Count |
| --- | ---: |
| no | 12597 |
| yes | 8578 |
| tomato | 2328 |
| diagnosis: huanglongbing (hlb), or citrus greening | 1087 |
| the diagnosis is tomato yellow leaf curl virus (tylcv) | 1077 |
| the causal agent is the tomato yellow leaf curl virus | 1066 |
| the leaf shows blotchy, asymmetrical yellowing, a classic symptom of hlb | 1058 |
| this is a viral infection: tylcv | 1045 |
| the cause is the bacterium candidatus liberibacter asiaticus, leading to huanglongbing | 1044 |
| this is citrus greening disease | 1028 |
| this tomato leaf shows classic signs of tylcv, like yellowing and curling | 1011 |
| orange | 676 |
| the condition is identified as late blight | 590 |
| the large, dark, water-soaked lesions are a key sign of late blight | 578 |
| the plant is suffering from a late blight infection | 567 |
| this is late blight, caused by the oomycete phytophthora infestans | 554 |
| potato late blight evidence: visible potato symptoms support the late blight label | 536 |
| tomato late blight evidence: visible tomato symptoms support the late blight label | 536 |
| soybean healthy evidence: visible soybean symptoms support the healthy label | 536 |
| squash powdery mildew evidence: visible squash symptoms support the powdery mildew label | 536 |
| potato early blight evidence: visible potato symptoms support the early blight label | 536 |
| strawberry healthy evidence: visible strawberry symptoms support the healthy label | 536 |
| tomato early blight evidence: visible tomato symptoms support the early blight label | 536 |
| potato healthy evidence: visible potato symptoms support the healthy label | 536 |
| tomato healthy evidence: visible tomato symptoms support the healthy label | 536 |
| tomato target spot evidence: visible tomato symptoms support the target spot label | 512 |
| pepper bell healthy evidence: visible pepper, bell symptoms support the healthy label | 512 |
| tomato spider mites two spotted spider mite evidence: visible tomato symptoms support the spider mites two-spotted spider mite label | 512 |
| blueberry healthy evidence: visible blueberry symptoms support the healthy label | 512 |
| grape black rot evidence: visible grape symptoms support the black rot label | 512 |

## Examples: `short_answer`

| # | Dataset | Task | Sample ID | Target |
| ---: | --- | --- | --- | --- |
| 1 | plantvillage_vqa | vqa | `plantvillage_vqa-image_044320.JPG-191096` | Answer: Yes |
| 2 | plantvillage_vqa | vqa | `plantvillage_vqa-image_036202.JPG-106447` | Answer: No |
| 3 | plantvillage_vqa | vqa | `plantvillage_vqa-image_029028.JPG-085371` | Answer: Tomato |
| 4 | plantvillage_vqa | vqa | `plantvillage_vqa-image_042921.JPG-126190` | Answer: Yes |
| 5 | plantvillage_vqa | vqa | `plantvillage_vqa-image_026404.JPG-187055` | Answer: Yes |
| 6 | plantvillage_vqa | vqa | `plantvillage_vqa-image_031311.jpg-092156` | Answer: Corn |
| 7 | plantvillage_vqa | vqa | `plantvillage_vqa-image_003983.JPG-011592` | Answer: No |
| 8 | plantvillage_vqa | vqa | `plantvillage_vqa-image_025887.JPG-186935` | Answer: Yes |
| 9 | plantvillage_vqa | vqa | `plantvillage_vqa-image_009250.JPG-027063` | Answer: No |
| 10 | plantvillage_vqa | vqa | `plantvillage_vqa-image_034532.JPG-101653` | Answer: Tomato |
