# Ablation Configs

These configs are paper-facing variants. Run them only after the main SFT checkpoint is available.

Reward ablations:

- `rl_grpo_no_clarify_vs_respond.yaml`
- `rl_grpo_no_uncertainty_calibration.yaml`
- `rl_grpo_no_hallucination_penalty.yaml`
- `rl_grpo_no_management_coverage.yaml`

SFT ablations:

- `sft_lora_attention_only_full_l4.yaml`
- `sft_freeze_projector_full_l4.yaml`

Example:

```bash
PYTHONPATH=src python scripts/train/train_rl_grpo.py \
  --model-config configs/model/qwen_vlm_4b_l4.yaml \
  --train-config configs/ablations/rl_grpo_no_clarify_vs_respond.yaml
```

Additional planned ablations are tracked in `docs/experiment_roadmap.md`.
