# SFT Target Quality Audit

- Manifest: `data/manifests/full/sft_train_phi4_max3_balanced_v2_instructional_labelrepaired.jsonl`
- Rows: `180000`
- Target format: `instructional`

## Task Mix

| Task | Rows |
| --- | ---: |
| clarify_or_respond | 27000 |
| classification | 54000 |
| consultation | 36000 |
| vqa | 63000 |

## Target Lengths

| Field | Count | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| rendered_target_token_lengths | 180000 | 2 | 7.0 | 35.956 | 1155 |
| answer_token_lengths | 180000 | 1 | 6.0 | 35.306 | 1155 |

## Flags

| Flag | Count | Rate |
| --- | ---: | ---: |
| classification_numeric_label_prefix | 10 | 0.000056 |
| short_answer | 56739 | 0.315217 |

## Top Rendered Answers

| Answer | Count |
| --- | ---: |
| no | 15872 |
| yes | 10760 |
| tomato | 2968 |
| orange haunglongbing citrus greening | 2756 |
| tomato tomato yellow leaf curl virus | 2681 |
| soybean healthy | 2548 |
| cicadellidae | 1923 |
| lycorma delicatula | 1789 |
| miridae | 1739 |
| diagnosis: huanglongbing (hlb), or citrus greening | 1390 |
| aphids | 1387 |
| the diagnosis is tomato yellow leaf curl virus (tylcv) | 1344 |
| the leaf shows blotchy, asymmetrical yellowing, a classic symptom of hlb | 1343 |
| the cause is the bacterium candidatus liberibacter asiaticus, leading to huanglongbing | 1326 |
| this is a viral infection: tylcv | 1301 |
| the causal agent is the tomato yellow leaf curl virus | 1294 |
| this tomato leaf shows classic signs of tylcv, like yellowing and curling | 1268 |
| this is citrus greening disease | 1232 |
| peach bacterial spot | 1149 |
| tomato bacterial spot | 1071 |
| tomato septoria leaf spot | 969 |
| tomato late blight | 956 |
| orange | 946 |
| squash powdery mildew | 919 |
| tomato spider mites two-spotted spider mite | 839 |
| apple healthy | 836 |
| tomato healthy | 818 |
| the large, dark, water-soaked lesions are a key sign of late blight | 781 |
| blueberry healthy | 750 |
| pepper bell healthy | 740 |

## Examples: `short_answer`

| # | Dataset | Task | Sample ID | Target |
| ---: | --- | --- | --- | --- |
| 1 | plantvillage | classification | `plantvillage-train-043048` | Answer: tomato healthy |
| 2 | ip102 | classification | `ip102-ip102-v1-1-images-50469-jpg` | Answer: Xylotrechus |
| 3 | plantvillage_vqa | vqa | `plantvillage_vqa-image_015013.JPG-043920` | Answer: No |
| 4 | plantvillage_vqa | vqa | `plantvillage_vqa-image_040352.JPG-118734` | Answer: Potato |
| 5 | ip102 | classification | `ip102-ip102-v1-1-images-50394-jpg` | Answer: Xylotrechus |
| 6 | plantvillage_vqa | vqa | `plantvillage_vqa-image_002936.JPG-008542` | Answer: No |
| 7 | ip102 | classification | `ip102-ip102-v1-1-images-15931-jpg` | Answer: corn borer |
| 8 | ip102 | classification | `ip102-ip102-v1-1-images-72237-jpg` | Answer: Cicadellidae |
| 9 | plantvillage_vqa | vqa | `plantvillage_vqa-image_043053.JPG-126568` | Answer: No |
| 10 | plantvillage_vqa | vqa | `plantvillage_vqa-image_025689.JPG-075351` | Answer: Tomato |

## Examples: `classification_numeric_label_prefix`

| # | Dataset | Task | Sample ID | Target |
| ---: | --- | --- | --- | --- |
| 1 | agbase | consultation | `agbase-agbase-836959` | Diagnosis: 2,4-d type broadleaf herbicide damage<br>Evidence: Visible symptoms should be checked against the image and crop context.<br>Uncertainty: Image-only assessment; confirm with field context before treatment.<br>Management: No source-specific management step was provided; recommend local extension follow-up.<br>Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain. |
| 2 | agbase | consultation | `agbase-agbase-585016` | Diagnosis: 1. carrion plant 2. honeysuckle 3. black raspberry, virginia creeper<br>Evidence: Visible symptoms should be checked against the image and crop context.<br>Uncertainty: Image-only assessment; confirm with field context before treatment.<br>Management: No source-specific management step was provided; recommend local extension follow-up.<br>Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain. |
| 3 | agbase | consultation | `agbase-agbase-841118` | Diagnosis: 4 lined plant bug for honey crisp, likely fungal disease for crab apple<br>Evidence: Visible symptoms should be checked against the image and crop context.<br>Uncertainty: Image-only assessment; confirm with field context before treatment.<br>Management: no treatment needed for honey crisp tree as the damage is superficial and cosmetic. for crab apple, expert recommends sending sample to plant disease clinic for accurate diagnosis<br>Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain. |
| 4 | agbase | consultation | `agbase-agbase-649244` | Diagnosis: 4 lined plant bug damage and/or fungal disease<br>Evidence: significant damage, wilting<br>Uncertainty: Image-only assessment; confirm with field context before treatment.<br>Management: test the soil before watering. water under the leaves, not overhead. water in the morning to allow leaves to dry, and again late in the day if the soil is dry to 2 inches.<br>Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain. |
| 5 | agbase | consultation | `agbase-agbase-658447` | Diagnosis: 1. smartweed or knotweed (persicaria), 2. cabbage family (possibly brassica oleracea), 3. unknown (possibly flowering cherry or a root sucker from a nearby tree)<br>Evidence: Visible symptoms should be checked against the image and crop context.<br>Uncertainty: Image-only assessment; confirm with field context before treatment.<br>Management: No source-specific management step was provided; recommend local extension follow-up.<br>Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain. |
| 6 | agrillava | consultation | `agrillava-agrillava-000043` | Diagnosis: 12 days<br>Evidence: Visible symptoms should be checked against the image and crop context.<br>Uncertainty: Image-only assessment; confirm with field context before treatment.<br>Management: No source-specific management step was provided; recommend local extension follow-up.<br>Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain. |
| 7 | agbase | consultation | `agbase-agbase-723809` | Diagnosis: 1. unknown plant (possibly false daisy), 2. purple beautyberry (callicarpa)<br>Evidence: Visible symptoms should be checked against the image and crop context.<br>Uncertainty: Image-only assessment; confirm with field context before treatment.<br>Management: No source-specific management step was provided; recommend local extension follow-up.<br>Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain. |
| 8 | agrillava | consultation | `agrillava-agrillava-000404` | Diagnosis: 8.0-9.8mm<br>Evidence: Visible symptoms should be checked against the image and crop context.<br>Uncertainty: Image-only assessment; confirm with field context before treatment.<br>Management: No source-specific management step was provided; recommend local extension follow-up.<br>Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain. |
| 9 | agbase | consultation | `agbase-agbase-756931` | Diagnosis: 4 o'clocks<br>Evidence: Visible symptoms should be checked against the image and crop context.<br>Uncertainty: Image-only assessment; confirm with field context before treatment.<br>Management: remove most of the larvae by hand, or apply bt (bacillus thuringiensis) to exposed larvae<br>Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain. |
| 10 | agbase | consultation | `agbase-agbase-753624` | Diagnosis: 2,4-d damage from herbicide<br>Evidence: shriveling, yellowing on leaves at their bases<br>Uncertainty: Image-only assessment; confirm with field context before treatment.<br>Management: wait and see, cut back damaged areas when cooler, avoid repeat herbicide application in hot weather<br>Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain. |
