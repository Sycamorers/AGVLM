# Training Plan

## SFT

Default SFT path:
- base model: `Qwen/Qwen3-VL-4B-Instruct`
- optimizer path: LoRA / QLoRA-style
- freeze the vision encoder by default
- keep projector freezing configurable
- mixed-task batching across classification, VQA, consultation, and clarify/respond samples

Why LoRA by default:
- smaller GPU footprint
- faster iteration for V1
- lower operational risk than full fine-tuning

Optional path:
- full fine-tuning through `configs/train/sft_full_optional.yaml`

## RL post-training

Default RL path:
- start from the SFT checkpoint
- use `trl.GRPOTrainer`
- default `loss_type: grpo`

Configurable GRPO-family variants:
- `dr_grpo`
- `dapo`
- other TRL-supported loss options through config changes

Why keep GRPO as the default:
- matches the requested V1 objective
- integrates with mainstream TRL tooling
- keeps the repo maintainable without custom RL infrastructure

## Runtime notes

Recommended hardware assumptions:
- one modern CUDA GPU for smoke-sized experiments
- more GPU memory for 8B or larger completion lengths

Optional stack features:
- FlashAttention-2 for faster memory-efficient attention
- DeepSpeed for larger runs
- vLLM for faster RL rollout generation when the environment is stable
