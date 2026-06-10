# Classification Error Analysis

This report samples failed classification examples after normalized parsing.

## Failure categories in sampled errors

| error_category | count_in_sample |
| --- | --- |
| true semantic error or source-level prediction collapse | 128 |
| synonym/canonical label mismatch or out-of-space output | 22 |

## Failed examples

| run | sample_id | source_dataset | reference | prediction | parse_status | error_category | raw_output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stage5 | plantvillage-train-028272 | plantvillage | strawberry leaf scorch | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | banana_disease-train-000607 | banana_disease | yellow sigatoka | black sigatoka | exact | true semantic error or source-level prediction collapse | Answer: black sigatoka |
| stage5 | rice_disease-train-003334 | rice_disease | healthy | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | ip102-ip102-v1-1-images-13040-jpg | ip102 | large cutworm | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | ip102-ip102-v1-1-images-18053-jpg | ip102 | aphids | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | ip102-ip102-v1-1-images-25631-jpg | ip102 | wheat sawfly | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | tea_sickness-validation-000079 | tea_sickness | white spot | gray blight | out_of_label_space | synonym/canonical label mismatch or out-of-space output | Answer: gray blight Evidence: The image shows symptoms of gray blight, which is a fungal disease that affects crops. |
| stage5 | plantdoc-1355 | plantdoc | corn gray leaf spot | bell pepper leaf spot | exact | true semantic error or source-level prediction collapse | Answer: bell pepper leaf spot |
| stage5 | plantvillage-train-003533 | plantvillage | blueberry healthy | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | plantvillage-train-041459 | plantvillage | tomato tomato yellow leaf curl virus | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | plantvillage-train-023084 | plantvillage | soybean healthy | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | ip102-ip102-v1-1-images-19465-jpg | ip102 | aphids | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | plantdoc-6147 | plantdoc | tomato septoria leaf spot | bell pepper leaf spot | exact | true semantic error or source-level prediction collapse | Answer: bell pepper leaf spot |
| stage5 | banana_disease-train-000384 | banana_disease | insect pest | black sigatoka | exact | true semantic error or source-level prediction collapse | Answer: black sigatoka |
| stage5 | plantvillage-train-035348 | plantvillage | tomato spider mites two spotted spider mite | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | plantdoc-4730 | plantdoc | tomato mold leaf | bell pepper leaf spot | exact | true semantic error or source-level prediction collapse | Answer: bell pepper leaf spot |
| stage5 | digigreen_crop_disease-train-000446 | digigreen_crop_disease | maize healthy | healthy | out_of_label_space | synonym/canonical label mismatch or out-of-space output | Answer: healthy Evidence: No visible symptoms |
| stage5 | ip102-ip102-v1-1-images-74174-jpg | ip102 | Cicadellidae | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | rice_disease-validation-003272 | rice_disease | narrow brown spot | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | rice_disease-train-012096 | rice_disease | narrow brown spot | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | plantvillage-train-012440 | plantvillage | orange haunglongbing citrus greening | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | rice_disease-validation-003007 | rice_disease | rice yellow stem borer | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | ip102-ip102-v1-1-images-58813-jpg | ip102 | Panonchus citri McGregor | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | plantvillage-train-020462 | plantvillage | potato early blight | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | plantvillage-train-000098 | plantvillage | apple apple scab | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | rice_disease-train-024260 | rice_disease | sheath blight | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | rice_disease-train-002055 | rice_disease | rice skipper | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | rice_disease-train-004896 | rice_disease | healthy | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | rice_disease-train-003386 | rice_disease | rice skipper | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | digigreen_crop_disease-train-000418 | digigreen_crop_disease | brinjal mites | healthy | out_of_label_space | synonym/canonical label mismatch or out-of-space output | Answer: healthy Evidence: No visible symptoms |
| stage5 | ip102-ip102-v1-1-images-56632-jpg | ip102 | Miridae | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | ip102-ip102-v1-1-images-37240-jpg | ip102 | blister beetle | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | digigreen_crop_disease-train-000655 | digigreen_crop_disease | coriander healthy | healthy | out_of_label_space | synonym/canonical label mismatch or out-of-space output | Answer: healthy Evidence: No visible symptoms |
| stage5 | rice_disease-train-009925 | rice_disease | sheath blight | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | ip102-ip102-v1-1-images-01161-jpg | ip102 | rice leaf caterpillar | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | rice_disease-train-002479 | rice_disease | thrips | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | plantvillage-train-031024 | plantvillage | tomato early blight | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | plantvillage-train-024445 | plantvillage | soybean healthy | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | plantdoc-7455 | plantdoc | tomato leaf late blight | bell pepper leaf spot | exact | true semantic error or source-level prediction collapse | Answer: bell pepper leaf spot |
| stage5 | plantvillage-train-006384 | plantvillage | corn maize common rust | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | ip102-ip102-v1-1-images-37572-jpg | ip102 | blister beetle | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | rice_disease-train-014854 | rice_disease | brown spot | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | ip102-ip102-v1-1-images-24894-jpg | ip102 | longlegged spider mite | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | plantvillage-train-015414 | plantvillage | orange haunglongbing citrus greening | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | plantvillage-train-001974 | plantvillage | apple healthy | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | plantvillage-train-041048 | plantvillage | tomato tomato yellow leaf curl virus | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage5 | plantdoc-4067 | plantdoc | tomato leaf bacterial spot | bell pepper leaf spot | exact | true semantic error or source-level prediction collapse | Answer: bell pepper leaf spot |
| stage5 | ip102-ip102-v1-1-images-38288-jpg | ip102 | blister beetle | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage5 | rice_disease-train-011768 | rice_disease | narrow brown spot | bacterial leaf blight | exact | true semantic error or source-level prediction collapse | Answer: bacterial leaf blight Evidence: leaf blight |
| stage5 | ip102-ip102-v1-1-images-25205-jpg | ip102 | wheat phloeothrips | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: |
| stage6_mc | plantvillage-train-028272 | plantvillage | strawberry leaf scorch | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | banana_disease-train-000607 | banana_disease | yellow sigatoka | black sigatoka | exact | true semantic error or source-level prediction collapse | Answer: black sigatoka Evidence: The image shows dark spots on the banana leaves, which is a common symptom of black sigatoka. |
| stage6_mc | rice_disease-train-003334 | rice_disease | healthy | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | ip102-ip102-v1-1-images-13040-jpg | ip102 | large cutworm | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | ip102-ip102-v1-1-images-18053-jpg | ip102 | aphids | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | ip102-ip102-v1-1-images-25631-jpg | ip102 | wheat sawfly | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | tea_sickness-validation-000079 | tea_sickness | white spot | gray light | exact | true semantic error or source-level prediction collapse | Answer: gray light |
| stage6_mc | plantdoc-1355 | plantdoc | corn gray leaf spot | bell pepper leaf spot | exact | true semantic error or source-level prediction collapse | Answer: bell pepper leaf spot Evidence: The image shows a pepper leaf with spots, which is characteristic of bell pepper leaf spot. |
| stage6_mc | plantvillage-train-003533 | plantvillage | blueberry healthy | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | plantvillage-train-041459 | plantvillage | tomato tomato yellow leaf curl virus | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | plantvillage-train-023084 | plantvillage | soybean healthy | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | ip102-ip102-v1-1-images-19465-jpg | ip102 | aphids | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | plantdoc-6147 | plantdoc | tomato septoria leaf spot | bell pepper leaf spot | exact | true semantic error or source-level prediction collapse | Answer: bell pepper leaf spot Evidence: The image shows a pepper leaf with spots, which is characteristic of bell pepper leaf spot. |
| stage6_mc | banana_disease-train-000384 | banana_disease | insect pest | black sigatoka | exact | true semantic error or source-level prediction collapse | Answer: black sigatoka Evidence: The image shows dark spots on the banana leaves, which is a common symptom of black sigatoka. |
| stage6_mc | plantvillage-train-035348 | plantvillage | tomato spider mites two spotted spider mite | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | plantdoc-4730 | plantdoc | tomato mold leaf | bell pepper leaf spot | exact | true semantic error or source-level prediction collapse | Answer: bell pepper leaf spot Evidence: The image shows a pepper leaf with spots, which is characteristic of bell pepper leaf spot. |
| stage6_mc | digigreen_crop_disease-train-000446 | digigreen_crop_disease | maize healthy | arhar aphids | exact | true semantic error or source-level prediction collapse | Answer: arhar aphids Evidence: aphids visible on the plant |
| stage6_mc | ip102-ip102-v1-1-images-74174-jpg | ip102 | Cicadellidae | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | rice_disease-validation-003272 | rice_disease | narrow brown spot | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | rice_disease-train-012096 | rice_disease | narrow brown spot | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | plantvillage-train-012440 | plantvillage | orange haunglongbing citrus greening | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | rice_disease-validation-003007 | rice_disease | rice yellow stem borer | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | ip102-ip102-v1-1-images-58813-jpg | ip102 | Panonchus citri McGregor | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | plantvillage-train-020462 | plantvillage | potato early blight | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | plantvillage-train-000098 | plantvillage | apple apple scab | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | rice_disease-train-024260 | rice_disease | sheath blight | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | rice_disease-train-002055 | rice_disease | rice skipper | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | rice_disease-train-004896 | rice_disease | healthy | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | rice_disease-train-003386 | rice_disease | rice skipper | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | digigreen_crop_disease-train-000418 | digigreen_crop_disease | brinjal mites | arhar aphids | exact | true semantic error or source-level prediction collapse | Answer: arhar aphids Evidence: aphids visible on the plant |
| stage6_mc | ip102-ip102-v1-1-images-56632-jpg | ip102 | Miridae | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | ip102-ip102-v1-1-images-37240-jpg | ip102 | blister beetle | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | digigreen_crop_disease-train-000655 | digigreen_crop_disease | coriander healthy | arhar aphids | exact | true semantic error or source-level prediction collapse | Answer: arhar aphids Evidence: aphids visible on the plant |
| stage6_mc | rice_disease-train-009925 | rice_disease | sheath blight | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | ip102-ip102-v1-1-images-01161-jpg | ip102 | rice leaf caterpillar | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | rice_disease-train-002479 | rice_disease | thrips | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | plantvillage-train-031024 | plantvillage | tomato early blight | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | plantvillage-train-024445 | plantvillage | soybean healthy | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | plantdoc-7455 | plantdoc | tomato leaf late blight | bell pepper leaf spot | exact | true semantic error or source-level prediction collapse | Answer: bell pepper leaf spot Evidence: The image shows a pepper leaf with spots, which is characteristic of bell pepper leaf spot. |
| stage6_mc | plantvillage-train-006384 | plantvillage | corn maize common rust | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | ip102-ip102-v1-1-images-37572-jpg | ip102 | blister beetle | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | rice_disease-train-014854 | rice_disease | brown spot | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | ip102-ip102-v1-1-images-24894-jpg | ip102 | longlegged spider mite | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | plantvillage-train-015414 | plantvillage | orange haunglongbing citrus greening | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | plantvillage-train-001974 | plantvillage | apple healthy | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | plantvillage-train-041048 | plantvillage | tomato tomato yellow leaf curl virus | corn maize northern leaf blight | exact | true semantic error or source-level prediction collapse | Answer: corn maize northern leaf blight |
| stage6_mc | plantdoc-4067 | plantdoc | tomato leaf bacterial spot | bell pepper leaf spot | exact | true semantic error or source-level prediction collapse | Answer: bell pepper leaf spot Evidence: The image shows a pepper leaf with spots, which is characteristic of bell pepper leaf spot. |
| stage6_mc | ip102-ip102-v1-1-images-38288-jpg | ip102 | blister beetle | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage6_mc | rice_disease-train-011768 | rice_disease | narrow brown spot | rice gall midge | exact | true semantic error or source-level prediction collapse | Answer: rice gall midge |
| stage6_mc | ip102-ip102-v1-1-images-25205-jpg | ip102 | wheat phloeothrips | alfalfa weevil | exact | true semantic error or source-level prediction collapse | Answer: alfalfa weevil Evidence: alfalfa weevil |
| stage7_label_only_classification | plantvillage-train-028272 | plantvillage | strawberry leaf scorch | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | banana_disease-train-000607 | banana_disease | yellow sigatoka | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | rice_disease-train-003334 | rice_disease | healthy | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | plantvillage-train-007051 | plantvillage | corn maize northern leaf blight | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | ip102-ip102-v1-1-images-13040-jpg | ip102 | large cutworm | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | ip102-ip102-v1-1-images-25631-jpg | ip102 | wheat sawfly | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | tea_sickness-validation-000079 | tea_sickness | white spot | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | plantdoc-1355 | plantdoc | corn gray leaf spot | peach leaf | exact | true semantic error or source-level prediction collapse | peach leaf |
| stage7_label_only_classification | plantvillage-train-003533 | plantvillage | blueberry healthy | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | plantvillage-train-041459 | plantvillage | tomato tomato yellow leaf curl virus | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | plantvillage-train-023084 | plantvillage | soybean healthy | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | plantdoc-6147 | plantdoc | tomato septoria leaf spot | peach leaf | exact | true semantic error or source-level prediction collapse | peach leaf |
| stage7_label_only_classification | banana_disease-train-000384 | banana_disease | insect pest | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | plantvillage-train-035348 | plantvillage | tomato spider mites two spotted spider mite | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | plantdoc-4730 | plantdoc | tomato mold leaf | peach leaf | exact | true semantic error or source-level prediction collapse | peach leaf |
| stage7_label_only_classification | digigreen_crop_disease-train-000446 | digigreen_crop_disease | maize healthy | aphids | out_of_label_space | synonym/canonical label mismatch or out-of-space output | aphids |
| stage7_label_only_classification | ip102-ip102-v1-1-images-74174-jpg | ip102 | Cicadellidae | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | rice_disease-validation-003272 | rice_disease | narrow brown spot | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | rice_disease-train-012096 | rice_disease | narrow brown spot | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | plantvillage-train-012440 | plantvillage | orange haunglongbing citrus greening | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | rice_disease-validation-003007 | rice_disease | rice yellow stem borer | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | ip102-ip102-v1-1-images-58813-jpg | ip102 | Panonchus citri McGregor | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | plantvillage-train-020462 | plantvillage | potato early blight | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | plantvillage-train-000098 | plantvillage | apple apple scab | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | rice_disease-train-024260 | rice_disease | sheath blight | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | rice_disease-train-002055 | rice_disease | rice skipper | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | rice_disease-train-004896 | rice_disease | healthy | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | rice_disease-train-003386 | rice_disease | rice skipper | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | digigreen_crop_disease-train-000418 | digigreen_crop_disease | brinjal mites | aphids | out_of_label_space | synonym/canonical label mismatch or out-of-space output | aphids |
| stage7_label_only_classification | ip102-ip102-v1-1-images-56632-jpg | ip102 | Miridae | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | ip102-ip102-v1-1-images-37240-jpg | ip102 | blister beetle | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | digigreen_crop_disease-train-000655 | digigreen_crop_disease | coriander healthy | aphids | out_of_label_space | synonym/canonical label mismatch or out-of-space output | aphids |
| stage7_label_only_classification | rice_disease-train-009925 | rice_disease | sheath blight | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | ip102-ip102-v1-1-images-01161-jpg | ip102 | rice leaf caterpillar | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | rice_disease-train-002479 | rice_disease | thrips | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | plantvillage-train-031024 | plantvillage | tomato early blight | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | plantvillage-train-024445 | plantvillage | soybean healthy | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | plantdoc-7455 | plantdoc | tomato leaf late blight | peach leaf | exact | true semantic error or source-level prediction collapse | peach leaf |
| stage7_label_only_classification | plantvillage-train-006384 | plantvillage | corn maize common rust | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | ip102-ip102-v1-1-images-37572-jpg | ip102 | blister beetle | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | rice_disease-train-014854 | rice_disease | brown spot | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | ip102-ip102-v1-1-images-24894-jpg | ip102 | longlegged spider mite | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | plantvillage-train-015414 | plantvillage | orange haunglongbing citrus greening | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | plantvillage-train-001974 | plantvillage | apple healthy | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | plantvillage-train-041048 | plantvillage | tomato tomato yellow leaf curl virus | peach bacterial spot | exact | true semantic error or source-level prediction collapse | peach bacterial spot |
| stage7_label_only_classification | plantdoc-4067 | plantdoc | tomato leaf bacterial spot | peach leaf | exact | true semantic error or source-level prediction collapse | peach leaf |
| stage7_label_only_classification | ip102-ip102-v1-1-images-38288-jpg | ip102 | blister beetle | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | rice_disease-train-011768 | rice_disease | narrow brown spot | to spot | out_of_label_space | synonym/canonical label mismatch or out-of-space output | to spot |
| stage7_label_only_classification | ip102-ip102-v1-1-images-25205-jpg | ip102 | wheat phloeothrips | aphids | exact | true semantic error or source-level prediction collapse | aphids |
| stage7_label_only_classification | ip102-ip102-v1-1-images-56629-jpg | ip102 | Miridae | aphids | exact | true semantic error or source-level prediction collapse | aphids |

The category labels are heuristic and intended to guide manual inspection. They should not be used to inflate metrics.
