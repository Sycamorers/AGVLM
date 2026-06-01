# SFT Target Quality Audit

- Manifest: `data/manifests/full/sft_train_phi4_max3_closed_label_classification_repair.jsonl`
- Rows: `167498`
- Target format: `instructional`

## Task Mix

| Task | Rows |
| --- | ---: |
| clarify_or_respond | 6482 |
| classification | 86016 |
| consultation | 25000 |
| vqa | 50000 |

## Target Lengths

| Field | Count | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| rendered_target_token_lengths | 167498 | 2 | 13.0 | 25.241 | 835 |
| answer_token_lengths | 167498 | 1 | 12.0 | 24.429 | 835 |

## Flags

| Flag | Count | Rate |
| --- | ---: | ---: |
| short_answer | 25797 | 0.154014 |

## Top Rendered Answers

| Answer | Count |
| --- | ---: |
| no | 12579 |
| yes | 8551 |
| tomato | 2320 |
| diagnosis: huanglongbing (hlb), or citrus greening | 1081 |
| the causal agent is the tomato yellow leaf curl virus | 1081 |
| the leaf shows blotchy, asymmetrical yellowing, a classic symptom of hlb | 1074 |
| the diagnosis is tomato yellow leaf curl virus (tylcv) | 1059 |
| this is citrus greening disease | 1056 |
| this is a viral infection: tylcv | 1050 |
| this tomato leaf shows classic signs of tylcv, like yellowing and curling | 1033 |
| the cause is the bacterium candidatus liberibacter asiaticus, leading to huanglongbing | 1032 |
| orange | 713 |
| the large, dark, water-soaked lesions are a key sign of late blight | 581 |
| the condition is identified as late blight | 573 |
| the plant is suffering from a late blight infection | 556 |
| this is late blight, caused by the oomycete phytophthora infestans | 550 |
| oides decempunctata evidence: visible agricultural symptoms or pest features support this label | 512 |
| wheat sawfly evidence: visible agricultural symptoms or pest features support this label | 512 |
| strawberry leaf scorch evidence: visible strawberry symptoms support the leaf scorch label | 512 |
| tomato early blight evidence: visible tomato symptoms support the early blight label | 512 |
| tomato tomato mosaic virus evidence: visible tomato symptoms support the tomato mosaic virus label | 512 |
| tomato septoria leaf spot evidence: visible tomato symptoms support the septoria leaf spot label | 512 |
| white backed plant hopper evidence: visible agricultural symptoms or pest features support this label | 512 |
| dacus dorsalis(hendel) evidence: visible agricultural symptoms or pest features support this label | 512 |
| rhytidodera bowrinii white evidence: visible agricultural symptoms or pest features support this label | 512 |
| tomato spider mites two spotted spider mite evidence: visible tomato symptoms support the spider mites two-spotted spider mite label | 512 |
| rice shell pest evidence: visible agricultural symptoms or pest features support this label | 512 |
| red spider evidence: visible agricultural symptoms or pest features support this label | 512 |
| peach borer evidence: visible agricultural symptoms or pest features support this label | 512 |
| wheat blossom midge evidence: visible agricultural symptoms or pest features support this label | 512 |

## Examples: `short_answer`

| # | Dataset | Task | Sample ID | Target |
| ---: | --- | --- | --- | --- |
| 1 | plantvillage_vqa | vqa | `plantvillage_vqa-image_038061.JPG-111972` | Answer: Orange |
| 2 | plantvillage_vqa | vqa | `plantvillage_vqa-image_038187.JPG-189771` | Answer: Yes |
| 3 | plantvillage_vqa | vqa | `plantvillage_vqa-image_020230.jpg-059128` | Answer: Yes |
| 4 | plantvillage_vqa | vqa | `plantvillage_vqa-image_033225.JPG-174170` | Answer: No |
| 5 | plantvillage_vqa | vqa | `plantvillage_vqa-image_003631.JPG-010504` | Answer: Orange |
| 6 | plantvillage_vqa | vqa | `plantvillage_vqa-image_003032.JPG-164927` | Answer: No |
| 7 | plantvillage_vqa | vqa | `plantvillage_vqa-image_038159.JPG-189765` | Answer: Yes |
| 8 | plantvillage_vqa | vqa | `plantvillage_vqa-image_037385.JPG-109946` | Answer: No |
| 9 | plantvillage_vqa | vqa | `plantvillage_vqa-image_043091.JPG-163757` | Answer: No |
| 10 | plantvillage_vqa | vqa | `plantvillage_vqa-image_000626.JPG-001779` | Answer: Yes |
