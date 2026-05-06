# Training Plan

## SFT

Active SFT path:
- base model: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- initialization: retained balanced LoRA adapter at `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft`
- optimizer path: bf16 LoRA with DeepSpeed ZeRO-3
- freeze the vision encoder by default
- keep projector freezing configurable
- train on the full max-3-image manifest with train/eval image-group overlap checks

Why LoRA by default:
- smaller GPU footprint
- faster iteration for V1
- lower operational risk than full fine-tuning

Legacy Qwen path:
- `Qwen/Qwen3-VL-4B-Instruct` remains useful for smoke tests, baseline wiring, and smaller debug runs.
- It is not the current next-stage training target.

Optional path:
- full fine-tuning through `configs/train/sft_full_optional.yaml`

## Active B200 Launch Plan

The next run should start with the 100-step probe:

```bash
sbatch \
  --export=ALL,TRAIN_CONFIG=configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml \
  scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm
```

If the probe completes with no OOM, acceptable step time, and valid checkpoint writes, submit the full run:

```bash
sbatch scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm
```

Active files:
- data split config: `configs/data/sft_train_eval_llama4_max3.yaml`
- model config: `configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml`
- DeepSpeed config: `configs/deepspeed/zero3_lora_b200_no_offload.json`
- probe train config: `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml`
- full train config: `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced.yaml`
- Slurm wrapper: `scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm`

The Slurm wrapper rebuilds:

```text
data/manifests/full/sft_train_llama4_max3_no_eval_overlap.jsonl
data/manifests/full/sft_eval_llama4_max3_stratified512.jsonl
data/manifests/full/sft_train_eval_llama4_max3_summary.json
```

Do not resume from the AGBASE-disjoint continuation. That path degraded validation behavior and was removed from Orange during cleanup.

## Generation Evaluation

Large training configs must keep:

```yaml
eval_generation_metrics: false
prediction_loss_only: true
```

Run generation evaluation separately on selected checkpoints. The May 6 AGBASE-disjoint job reached step `500` but stalled after loss eval because inline distributed generation metrics were too expensive.

## RL Post-Training

Default RL path:
- start from the completed full max3 SFT checkpoint
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

GRPO remains blocked until the full max3 SFT checkpoint and post-SFT benchmark results exist.

## Runtime Notes

Recommended hardware assumptions:
- one modern CUDA GPU for smoke-sized experiments
- 4x B200 for the active Llama 4 Scout max3 path
- more GPU memory or lower image pixels if future configs increase completion length, image count, or trainable modules

Monitoring outputs:
- TensorBoard: `<run_dir>/tensorboard/`
- JSONL metrics: `<run_dir>/metrics/train_metrics.jsonl`
- compatibility metrics: `<run_dir>/metrics.jsonl`
- run metadata: `<run_dir>/run_metadata.json`
- resolved config: `<run_dir>/resolved_config.yaml`

Optional stack features:
- FlashAttention-2 for faster memory-efficient attention after CUDA 12.9.1 validation
- DeepSpeed for larger runs
- vLLM for faster RL rollout generation when the environment is stable

## Llama 4 Scout Access

The `meta-llama/Llama-4-Scout-17B-16E-Instruct` path is gated by the Llama 4 license on Hugging Face. Authenticate the account before submitting the Slurm jobs; the Llama 4 Slurm scripts set `AGRI_VLM_REQUIRED_MODEL_ACCESS` so `scripts/verify_environment.py` checks `config.json` access before launching distributed training.

Do not use bitsandbytes QLoRA with ZeRO-3 for this path. Transformers injects a `device_map` for 4-bit bitsandbytes loading, and `device_map` is incompatible with DeepSpeed ZeRO-3.
