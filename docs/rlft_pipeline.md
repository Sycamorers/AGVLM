# RLFT Pipeline

## Current Status

RLFT is prepared but blocked on completion of the Phi-4 SFT run. Do not run smoke or full GRPO until `sft_checkpoint_path` in the selected RL config is replaced with a real completed SFT checkpoint or adapter path.

## Build The RL Manifest

```bash
PYTHONPATH=src python scripts/data/build_rl_manifest.py \
  --download-mode full \
  --fraction 1.0
```

Equivalent Make target:

```bash
make rl-data-full
```

The default RL build config is `configs/data/rl_build.yaml`. It excludes `test`, allows reward-verifiable verifier modes, limits long target answers, and keeps the default V1 RL subset to one image per sample.

## Audit The Manifest

```bash
PYTHONPATH=src python scripts/data/audit_rl_manifest.py \
  --manifest-path data/manifests/full/rl_manifest.jsonl \
  --output-json outputs/rl/audit/full_rl_manifest_audit.json \
  --output-md outputs/rl/audit/full_rl_manifest_audit.md \
  --fail-on-critical
```

Equivalent Make target:

```bash
make rl-audit-full
```

Critical issues include no samples, duplicate sample IDs, missing image paths, nonexistent image paths, unsupported verifier modes, clarify rows without expected decisions, and rows with no applicable reward module.

## Run Reward Sanity Check

```bash
PYTHONPATH=src python scripts/train/rl_reward_sanity_check.py \
  --manifest-path data/manifests/full/rl_manifest.jsonl \
  --config configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_readiness.yaml \
  --output-json outputs/rl/audit/full_rl_reward_sanity.json \
  --output-md outputs/rl/audit/full_rl_reward_sanity.md \
  --max-samples 200
```

Equivalent Make target:

```bash
make rl-reward-check-full
```

This does not load a VLM. It scores synthetic candidate completions through the configured composite reward function.

## Run Dataset Format Check

```bash
PYTHONPATH=src python scripts/train/check_rl_dataset_format.py \
  --manifest-path data/manifests/full/rl_manifest.jsonl \
  --model-config configs/model/phi4_reasoning_vision_15b_turin_24g.yaml \
  --max-samples 8 \
  --output-json outputs/rl/audit/rl_dataset_format_check.json \
  --output-md outputs/rl/audit/rl_dataset_format_check.md
```

Equivalent Make target:

```bash
make rl-format-check-full
```

This checks prompt conversion, image path resolution, JSON payload columns, and reward function column compatibility. It does not load model weights. Add `--check-processor` only when processor download/cache access is intended.

## Run CPU-Safe Tests

```bash
PYTHONPATH=src pytest tests -q
```

RL-only target:

```bash
make test-rl
```

## Run Readiness Dry-Run

```bash
PYTHONPATH=src python scripts/train/train_rl_grpo.py \
  --model-config configs/model/phi4_reasoning_vision_15b_turin_24g.yaml \
  --train-config configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_readiness.yaml \
  --dry-run
```

Equivalent Make target:

```bash
make rl-phi4-readiness
```

The readiness config intentionally contains a placeholder SFT checkpoint and `dry_run: true`. It validates config and manifest plumbing and writes a dry-run summary without loading model weights.

## After SFT Completes

1. Identify the completed SFT checkpoint or adapter path.
2. Replace `<FINAL_SFT_CHECKPOINT_OR_ADAPTER>` in the smoke config:
   `configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_smoke_after_sft.yaml`.
3. Confirm the path exists and is not `microsoft/Phi-4-reasoning-vision-15B`.
4. Submit the smoke job:

```bash
sbatch \
  --export=ALL,TRAIN_CONFIG=configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_smoke_after_sft.yaml \
  scripts/hpc/run_rl_grpo_b200_4gpu_phi4_reasoning_vision_15b.slurm
```

The Slurm wrapper fails fast if the manifest or SFT checkpoint is invalid.

## After Smoke Passes

Replace the placeholder in the full config:
`configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_full_after_sft.yaml`.

Then submit:

```bash
sbatch \
  --export=ALL,TRAIN_CONFIG=configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_full_after_sft.yaml \
  scripts/hpc/run_rl_grpo_b200_4gpu_phi4_reasoning_vision_15b.slurm
```

## Outputs

- audit reports: `outputs/rl/audit/`
- readiness dry-run: `outputs/rl/readiness/grpo-phi4-reasoning-vision-15b-b200-4gpu/`
- smoke outputs: `outputs/rl/smoke/` and `/orange/hmedeiros/qinruoyao/agvlm/outputs/rl/smoke/`
- full outputs: `outputs/rl/full/` and `/orange/hmedeiros/qinruoyao/agvlm/outputs/rl/full/`
- TensorBoard logs: under the configured run output directory

## Troubleshooting

- Placeholder checkpoint error: replace `<FINAL_SFT_CHECKPOINT_OR_ADAPTER>` with a real completed SFT checkpoint or adapter path.
- Missing checkpoint error: wait for SFT completion or correct the path.
- Raw/base model error: RLFT cannot start from `microsoft/Phi-4-reasoning-vision-15B`.
- Audit image failures: rebuild or restage the dataset so manifest image paths resolve.
- Reward sanity anomalies: inspect examples where target answers do not beat empty output or all candidates get zero reward.
- Processor check failures: rerun without `--check-processor` for CPU-only format validation, then validate processor access separately on the training environment.
