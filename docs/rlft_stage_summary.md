# RLFT Stage Summary

## Status

RLFT code and data-readiness are prepared for Phi-4 GRPO, but formal RLFT training has not started.

Training blocker: the completed Phi-4 SFT checkpoint or adapter path is not yet available. The active SFT run is expected under `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu`, but RL configs still contain the required `<FINAL_SFT_CHECKPOINT_OR_ADAPTER>` placeholder.

Training submitted: NO.

## Files Added

- `configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_readiness.yaml`
- `configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_smoke_after_sft.yaml`
- `configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_full_after_sft.yaml`
- `scripts/data/audit_rl_manifest.py`
- `scripts/train/rl_reward_sanity_check.py`
- `scripts/train/check_rl_dataset_format.py`
- `scripts/hpc/run_rl_grpo_b200_4gpu_phi4_reasoning_vision_15b.slurm`
- `tests/test_rl_readiness_pipeline.py`
- `docs/rlft_design.md`
- `docs/rlft_pipeline.md`
- `docs/rlft_stage_summary.md`

## Files Modified

- `Makefile`
- `README.md`
- `docs/experiment_roadmap.md`
- `src/agri_vlm/data/builders.py`
- `src/agri_vlm/rewards/clarify_decision.py`
- `src/agri_vlm/training/rl_trainer.py`
- `tests/test_manifest_builders.py`
- `tests/test_model_factory.py`
- `tests/test_reward_functions.py`

## Tests Added Or Updated

- reward module and composite reward routing tests
- robust clarify decision extraction tests
- RL manifest audit CLI tests
- reward sanity CLI tests
- RL dataset-format CLI tests
- Phi-4 RL config loading tests
- SFT checkpoint safety tests
- Slurm static checks
- RL manifest duplicate-ID builder regression test
- torch-dependent model-factory tests now skip when `torch` is unavailable

## Commands Run

Local shell note: `python` was not on PATH, so equivalent `python3` commands were used.

```bash
PYTHONPATH=src python3 scripts/data/build_rl_manifest.py --download-mode full --fraction 1.0
```

Result: `data/manifests/full/rl_manifest.jsonl`, 270,498 rows, `max_images_per_sample=1`.

```bash
PYTHONPATH=src python3 scripts/data/audit_rl_manifest.py \
  --manifest-path data/manifests/full/rl_manifest.jsonl \
  --output-json outputs/rl/audit/full_rl_manifest_audit.json \
  --output-md outputs/rl/audit/full_rl_manifest_audit.md \
  --fail-on-critical
```

Result: passed, 270,498 samples, zero critical issues.

```bash
PYTHONPATH=src python3 scripts/train/rl_reward_sanity_check.py \
  --manifest-path data/manifests/full/rl_manifest.jsonl \
  --config configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_readiness.yaml \
  --output-json outputs/rl/audit/full_rl_reward_sanity.json \
  --output-md outputs/rl/audit/full_rl_reward_sanity.md \
  --max-samples 200
```

Result: passed, 200 sampled rows. Average target-answer reward was 1.6270 versus empty output 0.0000.

```bash
PYTHONPATH=src python3 scripts/train/check_rl_dataset_format.py \
  --manifest-path data/manifests/full/rl_manifest.jsonl \
  --model-config configs/model/phi4_reasoning_vision_15b_turin_24g.yaml \
  --max-samples 8 \
  --output-json outputs/rl/audit/rl_dataset_format_check.json \
  --output-md outputs/rl/audit/rl_dataset_format_check.md
```

Result: passed, 8 checked rows, no issues.

```bash
PYTHONPATH=src python3 scripts/train/train_rl_grpo.py \
  --model-config configs/model/phi4_reasoning_vision_15b_turin_24g.yaml \
  --train-config configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_readiness.yaml \
  --dry-run
```

Result: passed, 8 dry-run rows, no model weights loaded, placeholder SFT path allowed only because this was a dry-run.

```bash
PYTHONPATH=src pytest tests -q
```

Result: passed.
The lightweight shell does not have `torch`, so torch-dependent model-factory tests were skipped as optional dependency checks.

```bash
make test-rl
```

Result: passed, 21 RL-specific tests.

## Report Paths

- audit JSON: `outputs/rl/audit/full_rl_manifest_audit.json`
- audit Markdown: `outputs/rl/audit/full_rl_manifest_audit.md`
- reward sanity JSON: `outputs/rl/audit/full_rl_reward_sanity.json`
- reward sanity Markdown: `outputs/rl/audit/full_rl_reward_sanity.md`
- dataset-format JSON: `outputs/rl/audit/rl_dataset_format_check.json`
- dataset-format Markdown: `outputs/rl/audit/rl_dataset_format_check.md`
- readiness dry-run summary: `outputs/rl/readiness/grpo-phi4-reasoning-vision-15b-b200-4gpu/dry_run_summary.json`

## Future Commands

After SFT completes, replace `<FINAL_SFT_CHECKPOINT_OR_ADAPTER>` with a real completed SFT checkpoint or adapter path in the smoke config, then run:

```bash
sbatch \
  --export=ALL,TRAIN_CONFIG=configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_smoke_after_sft.yaml \
  scripts/hpc/run_rl_grpo_b200_4gpu_phi4_reasoning_vision_15b.slurm
```

After smoke passes and the full config placeholder is replaced, run:

```bash
sbatch \
  --export=ALL,TRAIN_CONFIG=configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_full_after_sft.yaml \
  scripts/hpc/run_rl_grpo_b200_4gpu_phi4_reasoning_vision_15b.slurm
```

Both future commands are expected to fail fast until the SFT checkpoint placeholder is replaced with an existing path that is not the raw/base model.

## Remaining TODOs

- Wait for SFT completion.
- Replace SFT checkpoint placeholders in smoke/full RL configs.
- Run smoke-after-SFT on 4 B200 GPUs.
- Inspect smoke logs and metrics before full GRPO.
- Run full 4x B200 GRPO only after smoke passes.
- Run post-RL before/after evaluation against the SFT checkpoint.
