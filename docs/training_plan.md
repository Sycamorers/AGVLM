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

Current project status:
- full data manifests are ready
- baseline inference on `local_holdout` has been run on a B200 GPU
- the active execution milestone is full-data SFT on L4 with `configs/train/sft_lora_full_l4_multigpu.yaml`
- the latest visible L4 job failed with CUDA OOM; rerun with the chunked-loss mitigation before starting GRPO

Monitoring outputs:
- TensorBoard: `<run_dir>/tensorboard/`
- JSONL metrics: `<run_dir>/metrics/train_metrics.jsonl`
- run metadata: `<run_dir>/run_metadata.json`
- resolved config: `<run_dir>/resolved_config.yaml`

Optional stack features:
- FlashAttention-2 for faster memory-efficient attention
- DeepSpeed for larger runs
- vLLM for faster RL rollout generation when the environment is stable

## Llama 4 Scout SFT path

The `meta-llama/Llama-4-Scout-17B-16E-Instruct` path is gated by the Llama 4 license on Hugging Face. Authenticate the account before submitting the Slurm jobs; the Llama 4 Slurm scripts set `AGRI_VLM_REQUIRED_MODEL_ACCESS` so `scripts/verify_environment.py` checks `config.json` access before launching distributed training.

Use the bf16 ZeRO-3 LoRA configs for this model:
- preflight: `configs/train/sft_lora_turin_16gpu_preflight_llama4_scout_zero3.yaml`
- full run: `configs/train/sft_lora_turin_16gpu_llama4_scout_zero3.yaml`

Do not use bitsandbytes QLoRA with ZeRO-3 for this path. Transformers injects a `device_map` for 4-bit bitsandbytes loading, and `device_map` is incompatible with DeepSpeed ZeRO-3.
