# Benchmark Split Report

- seed: `42`
- output directory: `benchmarks/vlm_baselines/splits_stage4_datafix`
- fallback enabled: `False`

## Phase Summary

| Phase | Val rows | Test rows | Duplicate IDs | Missing images | Sample-ID overlap | Group overlap | Public test rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sft_benchmark | 152 | 616 | 0 | 0 | 0 | 0 | 0 |

## sft_benchmark

- rows by split: `{'val': 152, 'test': 616}`
- rows by source dataset: `{'ip102': 273, 'plantvillage_vqa': 337, 'plantvillage': 105, 'mirage': 32, 'plantdoc': 21}`
- rows by task type: `{'classification': 399, 'vqa': 337, 'clarify_or_respond': 32}`
- multi-image distribution: `{'1': 741, '2': 11, '3': 16}`
- prompt leakage count: `1`
- ground-truth leakage count: `0`

### Example Rows

- `clarify_or_respond`: `[{'sample_id': 'mirage-#841042', 'source_dataset': 'mirage', 'benchmark_split': 'test', 'images': ['data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000527-01.png'], 'target_preview': '{\'acceptable_answers\': ["Could you provide more details about the moth\'s size, any distinct markings, or patterns? Additionally, knowing your location could help determine if this moth is native to yo'}, {'sample_id': 'mirage-#878521', 'source_dataset': 'mirage', 'benchmark_split': 'test', 'images': ['data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000849-01.png', 'data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000849-02.png'], 'target_preview': "{'acceptable_answers': ['Can you confirm if the black dots in the images appear to be frass (caterpillar droppings) or something else? Additionally, do you notice any insects or larvae on the plants, "}, {'sample_id': 'mirage-#837063', 'source_dataset': 'mirage', 'benchmark_split': 'test', 'images': ['data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000242-01.png', 'data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000242-02.png', 'data/raw/mirage/full/images/MMMT_Direct/dev/MMMT_Direct-dev-000242-03.png'], 'target_preview': "{'acceptable_answers': ['Based on the information provided and the opinion from the Natorp expert, it seems likely that your boxwoods are suffering from winter damage. This can occur when plants are e"}]`
- `classification`: `[{'sample_id': 'ip102-ip102-v1-1-images-04223-jpg', 'source_dataset': 'ip102', 'benchmark_split': 'val', 'images': ['data/raw/ip102/full/ip102_v1.1/images/04223.jpg'], 'target_preview': "{'acceptable_answers': [], 'answer_text': 'Rice Stemfly', 'canonical_label': 'Rice Stemfly', 'canonical_labels': [], 'decision': None, 'structured': {}}"}, {'sample_id': 'ip102-ip102-v1-1-images-41018-jpg', 'source_dataset': 'ip102', 'benchmark_split': 'val', 'images': ['data/raw/ip102/full/ip102_v1.1/images/41018.jpg'], 'target_preview': "{'acceptable_answers': [], 'answer_text': 'Apolygus lucorum', 'canonical_label': 'Apolygus lucorum', 'canonical_labels': [], 'decision': None, 'structured': {}}"}, {'sample_id': 'ip102-ip102-v1-1-images-01381-jpg', 'source_dataset': 'ip102', 'benchmark_split': 'val', 'images': ['data/raw/ip102/full/ip102_v1.1/images/01381.jpg'], 'target_preview': "{'acceptable_answers': [], 'answer_text': 'rice leaf caterpillar', 'canonical_label': 'rice leaf caterpillar', 'canonical_labels': [], 'decision': None, 'structured': {}}"}]`
- `vqa`: `[{'sample_id': 'plantvillage_vqa-image_031224.JPG-091888', 'source_dataset': 'plantvillage_vqa', 'benchmark_split': 'test', 'images': ['data/raw/plantvillage_vqa/full/images/train/image_031224.JPG'], 'target_preview': "{'acceptable_answers': ['Yes'], 'answer_text': 'Yes', 'canonical_label': None, 'canonical_labels': [], 'decision': None, 'structured': {}}"}, {'sample_id': 'plantvillage_vqa-image_044189.JPG-129982', 'source_dataset': 'plantvillage_vqa', 'benchmark_split': 'test', 'images': ['data/raw/plantvillage_vqa/full/images/train/image_044189.JPG'], 'target_preview': "{'acceptable_answers': ['Yes'], 'answer_text': 'Yes', 'canonical_label': None, 'canonical_labels': [], 'decision': None, 'structured': {}}"}, {'sample_id': 'plantvillage_vqa-image_018736.JPG-054814', 'source_dataset': 'plantvillage_vqa', 'benchmark_split': 'test', 'images': ['data/raw/plantvillage_vqa/full/images/train/image_018736.JPG'], 'target_preview': "{'acceptable_answers': ['The numerous small, angular spots are characteristic of Bacterial Spot.'], 'answer_text': 'The numerous small, angular spots are characteristic of Bacterial Spot.', 'canonical"}]`
