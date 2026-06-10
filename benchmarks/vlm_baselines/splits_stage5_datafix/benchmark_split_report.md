# Benchmark Split Report

- seed: `42`
- output directory: `benchmarks/vlm_baselines/splits_stage5_datafix`
- fallback enabled: `False`

## Phase Summary

| Phase | Val rows | Test rows | Duplicate IDs | Missing images | Sample-ID overlap | Group overlap | Public test rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sft_benchmark | 288 | 736 | 0 | 0 | 0 | 0 | 0 |

## sft_benchmark

- rows by split: `{'val': 288, 'test': 736}`
- rows by source dataset: `{'rice_disease': 201, 'ip102': 275, 'tea_sickness': 34, 'plantvillage_vqa': 322, 'plantvillage': 104, 'banana_disease': 17, 'plantdoc': 21, 'mirage': 32, 'digigreen_crop_disease': 18}`
- rows by task type: `{'classification': 670, 'vqa': 322, 'clarify_or_respond': 32}`
- multi-image distribution: `{'1': 998, '3': 13, '2': 13}`
- prompt leakage count: `1`
- ground-truth leakage count: `0`

### Example Rows

- `clarify_or_respond`: `[{'sample_id': 'mirage-#831081', 'source_dataset': 'mirage', 'benchmark_split': 'test', 'images': ['data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000569-01.png', 'data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000569-02.png', 'data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000569-03.png'], 'target_preview': '{\'acceptable_answers\': ["The symptoms you\'re describing, such as the splitting bark and uneven leaf growth, could be indicative of a few potential issues. One possibility is that the tree is experienc'}, {'sample_id': 'mirage-#842770', 'source_dataset': 'mirage', 'benchmark_split': 'test', 'images': ['data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000711-01.png', 'data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000711-02.png'], 'target_preview': "{'acceptable_answers': ['Could you provide a description of the organisms or objects you found on the silver maple trunk, or any visible characteristics from the images you attached? This will help in"}, {'sample_id': 'mirage-#824137', 'source_dataset': 'mirage', 'benchmark_split': 'test', 'images': ['data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000782-01.png', 'data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000782-02.png'], 'target_preview': '{\'acceptable_answers\': ["Could you provide more details about the brown spots, such as their size, shape, and distribution on the leaves? Additionally, information about your camellias\' care routine, '}]`
- `classification`: `[{'sample_id': 'rice_disease-validation-004243', 'source_dataset': 'rice_disease', 'benchmark_split': 'val', 'images': ['data/raw/rice_disease/full/images/validation/004243.jpg'], 'target_preview': "{'acceptable_answers': [], 'answer_text': 'healthy', 'canonical_label': 'healthy', 'canonical_labels': [], 'decision': None, 'structured': {}}"}, {'sample_id': 'ip102-ip102-v1-1-images-38241-jpg', 'source_dataset': 'ip102', 'benchmark_split': 'val', 'images': ['data/raw/ip102/full/ip102_v1.1/images/38241.jpg'], 'target_preview': "{'acceptable_answers': [], 'answer_text': 'blister beetle', 'canonical_label': 'blister beetle', 'canonical_labels': [], 'decision': None, 'structured': {}}"}, {'sample_id': 'tea_sickness-validation-000002', 'source_dataset': 'tea_sickness', 'benchmark_split': 'val', 'images': ['data/raw/tea_sickness/full/images/validation/000002.jpg'], 'target_preview': "{'acceptable_answers': [], 'answer_text': 'white spot', 'canonical_label': 'white spot', 'canonical_labels': [], 'decision': None, 'structured': {}}"}]`
- `vqa`: `[{'sample_id': 'plantvillage_vqa-image_004230.JPG-012292', 'source_dataset': 'plantvillage_vqa', 'benchmark_split': 'test', 'images': ['data/raw/plantvillage_vqa/full/images/train/image_004230.JPG'], 'target_preview': "{'acceptable_answers': ['Tomato'], 'answer_text': 'Tomato', 'canonical_label': None, 'canonical_labels': [], 'decision': None, 'structured': {}}"}, {'sample_id': 'plantvillage_vqa-image_010135.JPG-029623', 'source_dataset': 'plantvillage_vqa', 'benchmark_split': 'test', 'images': ['data/raw/plantvillage_vqa/full/images/train/image_010135.JPG'], 'target_preview': "{'acceptable_answers': ['No'], 'answer_text': 'No', 'canonical_label': None, 'canonical_labels': [], 'decision': None, 'structured': {}}"}, {'sample_id': 'plantvillage_vqa-image_026390.JPG-077536', 'source_dataset': 'plantvillage_vqa', 'benchmark_split': 'test', 'images': ['data/raw/plantvillage_vqa/full/images/train/image_026390.JPG'], 'target_preview': "{'acceptable_answers': ['This is a healthy plant leaf.'], 'answer_text': 'This is a healthy plant leaf.', 'canonical_label': None, 'canonical_labels': [], 'decision': None, 'structured': {}}"}]`
