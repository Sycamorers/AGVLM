# Dataset and Project Scope Audit

## High-level diagnosis

The active project is post-training `microsoft/Phi-4-reasoning-vision-15B` for ground-level RGB agricultural consultation, classification, VQA, and clarify/respond behavior with LoRA SFT. The current Stage5 scope is broad: it mixes closed-label classification, short VQA, structured consultation, and dialogue-routing examples in one adapter. That is probably too heterogeneous for the available vertical classification signal, especially because classification has many source-specific label spaces and the benchmark shows source-level prediction collapse rather than small formatting drift.

## Split sizes

| split | num_samples |
| --- | --- |
| train | 143114 |
| val | 288 |
| test | 736 |

## Task-type mix

| split | task_type | num_samples |
| --- | --- | --- |
| test | clarify_or_respond | 32 |
| test | classification | 382 |
| test | vqa | 322 |
| train | clarify_or_respond | 6482 |
| train | classification | 61632 |
| train | consultation | 25000 |
| train | vqa | 50000 |
| val | classification | 288 |

## Task/domain summary

| task_name | task_type | num_train | num_val | num_test | num_classes | min_class_count | max_class_count | output_format | major_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plantvillage_vqa:vqa | vqa | 50000 | 0 | 322 |  |  |  | short_answer=50322 | needs_generation_or_qualitative_metrics |
| agbase:consultation | consultation | 20413 | 0 | 0 |  |  |  | structured_sections=20413 | needs_generation_or_qualitative_metrics |
| plantvillage:classification | classification | 19456 | 0 | 104 | 38 | 512 | 512 | bare_canonical_label_manifest=19560 | large_label_space |
| plantdoc:classification | classification | 14336 | 0 | 21 | 28 | 512 | 512 | bare_canonical_label_manifest=14357 | large_label_space |
| ip102:classification | classification | 13056 | 152 | 123 | 102 | 128 | 128 | bare_canonical_label_manifest=13331 | large_label_space |
| rice_disease:classification | classification | 8064 | 119 | 82 | 21 | 384 | 384 | bare_canonical_label_manifest=8265 | large_label_space |
| mirage:clarify_or_respond | clarify_or_respond | 6482 | 0 | 32 |  |  |  | decision_plus_answer_or_question=6514 | needs_generation_or_qualitative_metrics |
| digigreen_crop_disease:classification | classification | 5760 | 0 | 18 | 240 | 24 | 24 | bare_canonical_label_manifest=5778 | large_label_space; eval_labels_missing_from_train=4 |
| mirage:consultation | consultation | 3846 | 0 | 0 |  |  |  | structured_sections=3846 | needs_generation_or_qualitative_metrics |
| agrillava:consultation | consultation | 741 | 0 | 0 |  |  |  | structured_sections=741 | needs_generation_or_qualitative_metrics |
| tea_sickness:classification | classification | 512 | 17 | 17 | 8 | 64 | 64 | bare_canonical_label_manifest=546 | none_observed |
| banana_disease:classification | classification | 448 | 0 | 17 | 7 | 64 | 64 | bare_canonical_label_manifest=465 | none_observed |

Full table: `reports/task_distribution.csv`. Label table: `reports/label_distribution.csv`.

## Source-label metadata and synonym risks

The table below lists canonical labels that have multiple raw/source label strings attached in metadata. Some are true synonyms; others may be multi-label metadata or canonicalization collisions requiring manual review.

| task_name | label | variant_count | variants |
| --- | --- | --- | --- |
| digigreen_crop_disease:classification | tomato leaf curl virus | 5 | tomato fruit borer; tomato leaf curl virus; tomato nitrogen deficiency; tomato thrips; tomato whiteflies |
| digigreen_crop_disease:classification | brinjal phomopsis blight | 3 | brinjal mites; brinjal phomopsis blight; brinjal shoot and fruit borer |
| digigreen_crop_disease:classification | brinjal wilt | 3 | brinjal mites; brinjal shoot and fruit borer; brinjal wilt |
| digigreen_crop_disease:classification | cabbage alternaria leaf spot | 3 | cabbage alternaria leaf spot; cabbage cabbage looper; cabbage diamondback moth |
| digigreen_crop_disease:classification | chickpea botrytis gray mold | 3 | chickpea botrytis gray mold; chickpea dry root rot; chickpea fusarium wilt |
| digigreen_crop_disease:classification | chilli leaf curl virus | 3 | chilli leaf curl virus; chilli mites; chilli thrips |
| digigreen_crop_disease:classification | groundnut leaf spot | 3 | groundnut aphids; groundnut leaf miner; groundnut leaf spot |
| digigreen_crop_disease:classification | maize fall armyworm | 3 | maize fall armyworm; maize phosphorus deficiency; maize stem borer |
| digigreen_crop_disease:classification | mango anthracnose | 3 | mango anthracnose; mango boron deficiency; mango mango hopper |
| digigreen_crop_disease:classification | mango bacterial canker | 3 | mango bacterial canker; mango mango hopper; mango mealybugs |
| digigreen_crop_disease:classification | onion purple blotch | 3 | onion onion maggot; onion purple blotch; onion stemphylium blight |
| digigreen_crop_disease:classification | papaya papaya ring spot virus | 3 | papaya fruit fly; papaya papaya mealybug; papaya papaya ring spot virus |
| digigreen_crop_disease:classification | rice sheath blight | 3 | rice rice bug; rice sheath blight; rice thrips |
| digigreen_crop_disease:classification | tomato early blight | 3 | tomato early blight; tomato fruit borer; tomato thrips |
| plantdoc:classification | tomato early blight leaf | 3 | Potato leaf early blight; Tomato Early blight leaf; Tomato leaf |
| plantdoc:classification | tomato leaf | 3 | Tomato leaf; Tomato leaf bacterial spot; Tomato leaf late blight |
| plantdoc:classification | tomato two spotted spider mites leaf | 3 | Tomato Early blight leaf; Tomato leaf bacterial spot; Tomato two spotted spider mites leaf |
| digigreen_crop_disease:classification | banana panama wilt | 2 | banana banana weevil; banana panama wilt |
| digigreen_crop_disease:classification | banana sigatoka leaf spot | 2 | banana banana weevil; banana sigatoka leaf spot |
| digigreen_crop_disease:classification | bean bean beetle | 2 | bean bean beetle; bean pod borer |

## Duplicate and leakage checks

Exact split-overlap counts:

```json
{
  "duplicates_within_split": {
    "test": 263,
    "train": 105895,
    "val": 199
  },
  "image_group_overlap": {
    "train_test": 0,
    "train_val": 0,
    "val_test": 0
  },
  "prompt_target_hash_overlap": {
    "train_test": 410,
    "train_val": 89,
    "val_test": 62
  },
  "sample_id_overlap": {
    "train_test": 0,
    "train_val": 0,
    "val_test": 0
  }
}
```

Interpretation: sample-id and image-group overlap should be zero for train/test. Prompt-target hash overlap can be nonzero for repeated generic prompts and repeated labels, so it is a duplicate-risk signal rather than proof of leakage.

## Adapter-scope assessment

The dataset is not a clean fit for one general LoRA adapter if classification accuracy is a primary KPI. A single adapter may remain useful for agriculture consultation style and short VQA, but classification should be evaluated and trained as a separate track until label-only behavior and per-source label-space selection are stable.
