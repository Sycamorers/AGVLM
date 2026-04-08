# Decision Log

## 2026-04-08 Audit

Already working:
- dataset normalization, manifest building, evaluation scripts, and smoke/unit test scaffolding
- SFT entrypoint built on `transformers.Trainer`
- RL entrypoint built on `trl.GRPOTrainer`
- base-model selection already documented around `Qwen/Qwen3-VL-4B-Instruct`

Missing before this pass:
- no explicit multi-GPU launch path
- `device_map="auto"` made distributed training unsafe
- no rank-aware logging or artifact handling
- weak environment verification
- README was too long and did not standardize CUDA / Python / launch assumptions

## 2026-04-08 This Upgrade Pass

Changed:
- standardized the repo on Python `3.11`
- standardized setup around PyTorch `2.8.0` `cu129` wheels and CUDA 12.9.1 assumptions
- added `scripts/launch_torchrun.py` as the primary distributed launcher
- added distributed runtime helpers for rank, device, and logging behavior
- patched SFT and RL paths for distributed-safe model loading, main-rank-only artifact writes, and resume control
- added B200-oriented multi-GPU train configs and 1-GPU smoke configs
- rewrote `README.md`
- added explicit top-level `TODO.md`
- upgraded `scripts/verify_environment.py` to report CUDA, GPUs, bf16, and distributed sanity

## 2026-04-08 Deferred

Deferred on purpose:
- `flash-attn` is still optional until validated on the target CUDA 12.9.1 / B200 environment
- no Dockerfile was added because the repository did not already have one and the exact image choice still needs hardware validation
- no second distributed stack was added; `torchrun` is the single primary path for now
- real B200 multi-GPU validation remains a TODO because that hardware is not available in the current execution environment

## External Verification Used

- PyTorch previous versions page for official `cu129` wheel install instructions:
  - https://pytorch.org/get-started/previous-versions/
- PyTorch 2.7 release notes for Blackwell support context:
  - https://pytorch.org/blog/pytorch2-7/
- Transformers Qwen3-VL docs:
  - https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl
- TRL GRPO docs:
  - https://huggingface.co/docs/trl/grpo_trainer
