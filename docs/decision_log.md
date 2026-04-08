# Decision Log

## 2026-04-08: base VLM selection

Decision:
- default base model: `Qwen/Qwen3-VL-4B-Instruct`
- optional larger model: `Qwen/Qwen3-VL-8B-Instruct`

Why:
- the official Qwen Hugging Face organization exposes public open-weight `Qwen3-VL-4B-Instruct` and `Qwen3-VL-8B-Instruct` checkpoints
- the current official Hugging Face Transformers documentation includes `Qwen3-VL` in the v5.5.0 docs
- the official Qwen3-VL model cards recommend `flash_attention_2` where supported
- the default V1 path should use the smallest practical dense model

Inference explicitly noted:
- I could not verify an official public open-weight locally fine-tunable `Qwen3.6` VLM checkpoint on the official Qwen Hugging Face organization or current Transformers docs as of April 8, 2026
- because that is an absence-of-evidence conclusion from official sources, it is treated as an inference rather than a direct official statement

Sources:
- https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- https://huggingface.co/docs/trl/grpo_trainer

## 2026-04-08: training stack choice

Decision:
- `transformers`, `accelerate`, `peft`, `trl`, `datasets`, and `bitsandbytes` are the default stack
- `trl.GRPOTrainer` is used for RL
- LoRA is the default training path

Why:
- the stack is mainstream and maintainable
- TRL now documents VLM GRPO support and GRPO-family loss variants
- LoRA keeps V1 practical on limited hardware
