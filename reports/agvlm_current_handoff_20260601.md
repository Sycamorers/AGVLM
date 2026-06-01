# AGVLM Current Handoff

Date: 2026-06-01

Branch: `codex/sft-classification-repair-stage2-b200`

Scope: V1 remains limited to ground-level RGB agricultural consultation, including crop or pest classification, short visual QA, clarify/respond decisions, and structured agricultural advice. This branch does not make generic all-purpose VLM behavior the default path.

## Executive Status

The latest completed decision artifact promotes the B200 stage2 classification-repair SFT candidate over the active previous SFT according to the current SFT benchmark gate.

| Metric | Active previous SFT | Stage2 B200 candidate | Preferred delta |
| --- | ---: | ---: | ---: |
| Task macro average | `0.229` | `0.302` | `+0.073` |
| Short VQA relaxed accuracy | `0.156` | `0.224` | `+0.068` |
| Clarify/respond macro F1 | `0.530` | `0.681` | `+0.151` |
| Invalid predictions | `119` | `0` | `119` fewer |

Source artifact: `reports/sft_stage_decision_retry_full_20260601/summary.md`.

The stage2 candidate is therefore the current promotion candidate, but the branch also prepares a stage3 closed-label classification-repair run to reduce remaining label-space issues before any downstream RL/GRPO start. Stage3 is not yet the active checkpoint and must be benchmarked before replacing the promoted stage2 candidate.

## Completed Work

- Repaired SFT save validation so final PEFT adapter outputs are checked for non-empty LoRA tensors and written with `adapter_validation.json`.
- Added reusable checkpoint artifact validation for SFT, RL, and benchmark checkpoint configs so empty adapter files are not accepted silently.
- Added an `output_format` reward penalty and enabled it in Phi-4 GRPO configs and the reward sanity script. It penalizes blank `Answer:`, generic agricultural answers, missing clarify decisions, incomplete consultation sections, and runaway repetition.
- Added classification-repair prompt and target formatting for instructional SFT, including explicit nonblank `Answer:` and `Evidence:` requirements.
- Added closed-label classification label-space support in both SFT prompt construction and benchmark prompt construction.
- Added a closed-label, per-class-balanced SFT manifest builder and configs:
  - `scripts/data/build_closed_label_sft_manifest.py`
  - `configs/data/sft_closed_label_classification_repair_phi4_max3.yaml`
  - `configs/data/sft_format_audit_closed_label_classification_repair_phi4_max3.yaml`
- Added stage3 B200 configs initialized from the stage2 B200 adapter:
  - `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_closed_label_classification_repair_stage3_preflight.yaml`
  - `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_closed_label_classification_repair_stage3.yaml`
- Added benchmark support for the pending stage3 candidate in `benchmarks/vlm_baselines/agvlm_checkpoint_models.yaml`.
- Improved benchmark parsing and metrics so existing prediction artifacts can be rescored after parser fixes, and classification diagnostics now include accepted-label accuracy, semantic-alias accuracy, and out-of-label-space rate.
- Added Phi-4 / Phi-4-MM benchmark loader compatibility patches for current Transformers and PEFT behavior, plus safer Slurm cache locations and `MIN_NEW_TOKENS` control.
- Added static benchmark/dashboard tooling for SFT stage decisions and external baseline report generation.
- Generated current decision reports:
  - `reports/sft_stage_decision_20260601/summary.md`
  - `reports/sft_stage_decision_retry_full_20260601/summary.md`
  - `reports/sft_stage_decision_retry_full_20260601/dashboard.html`

## Current Work Underway

- Stage2 B200 is the promotion candidate from the latest retry-full benchmark decision, but operational promotion still needs a deliberate checkpoint/config handoff.
- Stage3 closed-label classification repair is prepared as the next training attempt. Its goal is to continue from stage2 and constrain classification outputs to source-specific allowed label sets.
- External baseline reporting is scaffolded through `benchmarks/vlm_baselines/slurm/run_external_baseline_report_array.sbatch` and should be rerun when comparing the promoted AGVLM candidate against public VLM baselines.
- RL/GRPO should remain gated. Do not start RL from any SFT checkpoint until the chosen SFT candidate has a completed benchmark report, non-empty adapter validation, and an explicit promotion decision.

## Next Steps

1. Confirm the stage2 adapter directory at `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-classification-repair-instructional-stage2-b200-4gpu` still contains a valid `adapter_validation.json` and non-empty adapter tensors.
2. Decide whether to promote stage2 immediately or run the prepared stage3 closed-label B200 SFT first.
3. If promoting stage2, update downstream active SFT references and keep the previous SFT path documented as the rollback checkpoint.
4. If running stage3, submit the stage3 preflight and full B200 config, then validate the saved adapter before benchmarking.
5. Benchmark the chosen candidate with the same SFT benchmark split and promotion gate used for stage2 retry-full.
6. Regenerate the SFT stage dashboard and keep the new `summary.md`, `summary.json`, and `dashboard.html` under a dated `reports/sft_stage_decision_*` directory.
7. Only after a candidate is promoted, run the external baseline matrix and then decide whether RL/GRPO readiness should proceed.

## Validation Already Recorded

- Earlier B200 stage2 preparation recorded a full project test pass: `143 passed, 2 warnings`.
- The stage2 retry-full benchmark decision completed with `Decision: PROMOTE`.
- The retry-full decision recorded `0` invalid predictions over `392` SFT benchmark examples for the stage2 candidate.

## Important Caveats

- Stage2 still had `19/392` parseable out-of-label answers in the decision diagnostics, which is why the closed-label stage3 path exists.
- Reports and benchmark artifacts in this repository are snapshots. Any new parser, prompt, or label-space change should rerun the relevant metric/report generation instead of treating older generated metrics as automatically current.
- Gated or licensed dataset steps must stay explicit in config and documentation; do not silently skip missing datasets, splits, adapters, or reward modules.
