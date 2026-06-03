# SFT Target Quality Audit

- Manifest: `data/manifests/full/sft_train_phi4_max3_stage4_closed_label_datafix.jsonl`
- Rows: `128330`
- Target format: `instructional`

## Task Mix

| Task | Rows |
| --- | ---: |
| clarify_or_respond | 6482 |
| classification | 46848 |
| consultation | 25000 |
| vqa | 50000 |

## Target Lengths

| Field | Count | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| rendered_target_token_lengths | 128330 | 2 | 13.0 | 28.813 | 1022 |
| answer_token_lengths | 128330 | 1 | 12.0 | 28.059 | 1022 |

## Flags

| Flag | Count | Rate |
| --- | ---: | ---: |
| short_answer | 25831 | 0.201286 |

## Top Rendered Answers

| Answer | Count |
| --- | ---: |
| no | 12593 |
| yes | 8629 |
| tomato | 2283 |
| diagnosis: huanglongbing (hlb), or citrus greening | 1117 |
| this is citrus greening disease | 1096 |
| the leaf shows blotchy, asymmetrical yellowing, a classic symptom of hlb | 1076 |
| the causal agent is the tomato yellow leaf curl virus | 1072 |
| the diagnosis is tomato yellow leaf curl virus (tylcv) | 1063 |
| the cause is the bacterium candidatus liberibacter asiaticus, leading to huanglongbing | 1036 |
| this tomato leaf shows classic signs of tylcv, like yellowing and curling | 1033 |
| this is a viral infection: tylcv | 985 |
| orange | 718 |
| the large, dark, water-soaked lesions are a key sign of late blight | 602 |
| the plant is suffering from a late blight infection | 576 |
| this is late blight, caused by the oomycete phytophthora infestans | 563 |
| the condition is identified as late blight | 560 |
| this is bacterial spot. note the small, dark, water-soaked lesions | 522 |
| the numerous small, angular spots are characteristic of bacterial spot | 518 |
| squash powdery mildew leaf evidence: visible agricultural symptoms or pest features support this label | 512 |
| tomato septoria leaf spot evidence: visible tomato symptoms support the septoria leaf spot label | 512 |
| tomato leaf late blight evidence: visible agricultural symptoms or pest features support this label | 512 |
| grape esca black measles evidence: visible grape symptoms support the esca (black measles) label | 512 |
| tomato healthy evidence: visible tomato symptoms support the healthy label | 512 |
| grape leaf blight isariopsis leaf spot evidence: visible grape symptoms support the leaf blight (isariopsis leaf spot) label | 512 |
| tomato early blight evidence: visible tomato symptoms support the early blight label | 512 |
| strawberry leaf scorch evidence: visible strawberry symptoms support the leaf scorch label | 512 |
| pepper bell bacterial spot evidence: visible pepper, bell symptoms support the bacterial spot label | 512 |
| tomato leaf mold evidence: visible tomato symptoms support the leaf mold label | 512 |
| potato leaf early blight evidence: visible agricultural symptoms or pest features support this label | 512 |
| corn maize healthy evidence: visible corn (maize) symptoms support the healthy label | 512 |

## Examples: `short_answer`

| # | Dataset | Task | Sample ID | Target |
| ---: | --- | --- | --- | --- |
| 1 | plantvillage_vqa | vqa | `plantvillage_vqa-image_001889.JPG-005535` | Answer: No |
| 2 | plantvillage_vqa | vqa | `plantvillage_vqa-image_035998.JPG-105818` | Answer: Leaf scorch |
| 3 | plantvillage_vqa | vqa | `plantvillage_vqa-image_014729.JPG-043071` | Answer: No |
| 4 | plantvillage_vqa | vqa | `plantvillage_vqa-image_012230.JPG-176059` | Answer: No |
| 5 | plantvillage_vqa | vqa | `plantvillage_vqa-image_036017.JPG-175920` | Answer: No |
| 6 | plantvillage_vqa | vqa | `plantvillage_vqa-image_024259.JPG-071107` | Answer: No |
| 7 | plantvillage_vqa | vqa | `plantvillage_vqa-image_034439.JPG-101360` | Answer: No |
| 8 | plantvillage_vqa | vqa | `plantvillage_vqa-image_012366.JPG-036116` | Answer: No |
| 9 | plantvillage_vqa | vqa | `plantvillage_vqa-image_042464.JPG-171786` | Answer: No |
| 10 | plantvillage_vqa | vqa | `plantvillage_vqa-image_036516.JPG-107368` | Answer: No |
