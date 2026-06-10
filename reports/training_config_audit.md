# Training Configuration Audit

## Config summary

| config | base_model | lora_r | alpha | dropout | lr | batch | grad_accum | max_steps | warmup | precision | eval_steps | pred_loss_only | gen_metrics | last_eval_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sft_phi4_reasoning_vision_15b_b200_4gpu_stage5_datafix.yaml | microsoft/Phi-4-reasoning-vision-15B | 256 | 512 | 0.0 | 2e-07 | 1 | 16 | 1000 | 0.03 | bf16 | 200 | True | False | 3.233602523803711 |
| sft_phi4_reasoning_vision_15b_b200_4gpu_classification_probe_stage6_mc.yaml | microsoft/Phi-4-reasoning-vision-15B | 256 | 512 | 0.0 | 1e-06 | 1 | 16 | 160 | 0.02 | bf16 | 20 | True | False | 1.4993938207626343 |
| sft_phi4_reasoning_vision_15b_b200_4gpu_stage7_label_only_classification.yaml | microsoft/Phi-4-reasoning-vision-15B | 256 | 512 | 0.0 | 5e-07 | 1 | 16 | 1000 | 0.03 | bf16 | 200 | True | True | 1.9396591186523438 |

## Assessment

- LoRA rank 256 / alpha 512 is high-capacity for a heterogeneous SFT adapter and can fit style/format without guaranteeing classification discrimination.
- Stage5 uses loss-only validation (`prediction_loss_only: true`, generation metrics disabled), so decreasing eval loss is not evidence that classification accuracy improved.
- Stage5 starts from an earlier classification-repair adapter but mixes classification, VQA, consultation, and clarify/respond again, creating task-interference risk.
- Stage6 MC is a useful probe but has only 280 train rows and 96 eval rows; it cannot be treated as a complete retraining fix.
- There is no configured early stopping; checkpoint choice should be tied to generation/evaluation metrics, not loss alone.
