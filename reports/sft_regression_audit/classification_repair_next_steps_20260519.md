# Classification Repair SFT Next Steps

## Why this round exists

The completed balanced-v2 SFT was not promoted because it regressed the benchmark gate. It improved VQA relaxed accuracy, but it produced more invalid predictions and lower clarify/respond F1 than the previous SFT. The largest hard failure is classification: base Phi-4, the previous SFT, and balanced-v2 all scored classification macro F1 `0.0` because all `114/114` classification benchmark predictions were invalid.

The invalid-output pattern is different by checkpoint. Base Phi-4 and the previous SFT mostly refused classification requests. Balanced-v2 instead collapsed to an empty `Answer:` response on every classification example. The repair round therefore targets output-format adherence and classification label supervision directly.

## What changed

- Updated instructional classification training prompts to match the benchmark output contract:
  - `Answer: <canonical agricultural label>`
  - `Evidence: <brief visible symptom evidence>`
- Updated instructional classification targets to include an `Evidence:` line after the canonical label.
- Built a classification-heavy SFT repair manifest from the non-overlapping max-3-image train split.
- Continued training from the previous best SFT adapter instead of starting again from raw Phi-4.
- Lowered learning rate to `1.0e-6` for a bounded repair pilot to reduce catastrophic drift.

## Repair Data

Manifest: `data/manifests/full/sft_train_phi4_max3_classification_repair_instructional.jsonl`

| Task | Rows |
| --- | ---: |
| classification | 86228 |
| vqa | 50000 |
| consultation | 25000 |
| clarify_or_respond | 6482 |
| total | 167710 |

The manifest includes all available classification rows and does not repeat rows. VQA and consultation are capped so classification becomes the dominant training signal while the model still sees non-classification agriculture tasks.

## Validation

- Targeted tests passed: `PYTHONPATH=src pytest -q tests/test_collators.py tests/test_manifest_builders.py tests/test_sft_trainer.py tests/test_benchmark_checkpoint_config.py`
- Train configs validated with `TrainConfigSchema`.
- Target-quality audit completed:
  - Report: `reports/sft_regression_audit/classification_repair_sft_target_quality_20260519.md`
  - JSON: `reports/sft_regression_audit/classification_repair_sft_target_quality_20260519.json`

Audit flags are expected for this dataset. IP102 labels start with numeric class IDs, and many VQA answers are intentionally short yes/no or crop-name targets.

## Submitted Training

Slurm job: `32682905`

Partition: `hpg-turin`

Status at launch check: `RUNNING` on 8 Turin nodes. Environment verification passed inside the job, and the preflight run directory was created.

Wrapper: `scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm`

Preflight config: `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_classification_repair_instructional_preflight.yaml`

Pilot config: `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_classification_repair_instructional_pilot.yaml`

Expected output after the wrapper adds batch suffix:

- Local: `outputs/sft/phi4-reasoning-vision-15b-classification-repair-instructional-pilot-turin16-batch1`
- Orange: `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-classification-repair-instructional-pilot-turin16-batch1`

## Success Criteria

The pilot should only be promoted if it fixes the failure mode without harming the tasks that were already acceptable. Minimum checks after completion:

- Classification invalid predictions must drop substantially from `114/114`.
- Classification macro F1 must become greater than `0.0`.
- Overall task macro should beat the previous active SFT baseline `0.228774`.
- Clarify/respond F1 should not regress below the previous active SFT baseline `0.530323`.
- Total invalid predictions should be below the previous active SFT count `119/392`.

## Next Actions

1. Watch job `32682905` through preflight and pilot completion.
2. If it fails, inspect `logs/slurm/agri-vlm-sft-phi4rv-16g-32682905.out` and `.err`, fix the launch or training issue, and resubmit.
3. If it completes, add the adapter to `benchmarks/vlm_baselines/agvlm_checkpoint_models.yaml`.
4. Run the same benchmark suite used for the previous SFT, balanced-v2 SFT, and raw Phi-4 base.
5. Run the promotion gate. Only use this pilot as the new active checkpoint if the gate passes.
