# RL/GRPO Preparation Report

- Repo commit at audit start: `683cfa0be3dd989d04fc136a58ac30c3fb4846b2`
- Scope: stage-2 rule-based GRPO initialized from a completed SFT checkpoint or adapter.
- SFT note: SFT-stage files were intentionally left untouched by this RL prep work. The worktree already had dirty SFT-related files before this task: `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_full_max3.yaml`, `scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm`, and related B200/checkpointing files. Those pre-existing changes were not reverted.

## What Changed

- Added RL-only data preparation: `scripts/data/prepare_rl_datasets.py`.
- Updated RL manifest construction to apply task-specific output formats, deterministic forbidden-claim metadata, normalized tuple deduplication, and a source/task-stratified RL holdout manifest.
- Strengthened RL rewards with parsed `Answer:`, `Decision:`, and true line-start section headers.
- Added conservative overconfidence, forbidden-claim, chemical-advice, length, and repetition penalties.
- Added RL config aliases: `configs/train/rl_grpo_phi4_readiness.yaml`, `configs/train/rl_grpo_phi4_smoke.yaml`, and `configs/train/rl_grpo_phi4_full.yaml`.
- Added RL-only Slurm entrypoints: `scripts/hpc/run_rl_grpo_phi4_smoke.slurm` and `scripts/hpc/run_rl_grpo_phi4_full.slurm`.
- Added RL checkpoint evaluation: `scripts/eval/eval_rl_checkpoint.py`.
- Added/updated RL tests, manifest audit, reward sanity, dataset format checks, and docs.

## Data Sources

Used in full RL preparation:

| Source | Role | Source rows | RL train rows | RL holdout rows |
| --- | --- | ---: | ---: | ---: |
| `plantvillage_vqa` | main short VQA | 193609 | 152678 | 2299 |
| `agbase` | manual AgMMU/AgBase-style consultation | 44849 | 13799 | 222 |
| `mirage` | agriculture consultation and clarify/respond | 40889 | 2975 | 77 |
| `plantvillage` | auxiliary classification | 54381 | 42846 | 657 |
| `ip102` | auxiliary pest classification | 75222 | 51812 | 791 |
| `plantdoc` | auxiliary field-style classification | 2578 | 2292 | 50 |

Unavailable/manual notes:

- `agbase` and `ip102` are manual/licensed-style sources but were already materialized locally for the full subset.
- `agrillava` remains excluded from RL config because licensing/download and verifiable conversion are not clear enough for the default RL path.
- No public `test` split is included in RL training.

## Manifest Counts

- RL train manifest: `data/manifests/full/rl_manifest.jsonl`
- RL local holdout manifest: `data/manifests/full/rl_local_holdout_eval.jsonl`
- Train rows: `266402`
- Holdout rows: `4096`
- Duplicate normalized rows removed: `0`

Train by task:

| Task | Rows |
| --- | ---: |
| `vqa` | 152678 |
| `classification` | 96950 |
| `consultation` | 15001 |
| `clarify_or_respond` | 1773 |

## Reward Changes

- `exact_match` compares the extracted `Answer:` field against accepted answers.
- `normalized_label` compares the extracted `Answer:` field against canonical and accepted labels.
- `structured_format` requires real line-start headers and non-empty section bodies.
- `clarify_vs_respond` prefers explicit `Decision:` parsing.
- `uncertainty_calibration` no longer treats `high confidence` or `confirm` as uncertainty markers.
- `management_coverage` remains a capped weak auxiliary reward.
- `hallucination_penalty` includes forbidden claims, overconfidence, unsupported chemical advice, length, and repetition penalties.

## Validation Results

Passed:

```bash
make rl-data-full
make rl-audit-full
make rl-reward-check-full
make rl-format-check-full
make rl-phi4-readiness
PYTHONPATH=src python3 scripts/train/train_rl_grpo.py --config configs/train/rl_grpo_phi4_readiness.yaml --dry-run
make test-rl
PYTHONPATH=src python3 scripts/eval/eval_rl_checkpoint.py --prediction-mode oracle --max-examples 10 --metrics-output reports/rl_eval_metrics.json --samples-output reports/rl_eval_samples.jsonl --report-output reports/rl_eval_report.md
```

Audit summary:

- Critical RL manifest issues: `0`
- Reward sanity assertion failures: `0`
- Format check issues: `0`
- Readiness transformed sample: one RGB image under the `images` key.

Expected guard failure:

```bash
make rl-phi4-smoke-after-sft
```

This failed before launch because `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1` exists but does not yet contain completed SFT model or adapter artifacts.

## Launch Commands

Readiness dry-run:

```bash
make rl-phi4-readiness
```

Smoke after SFT checkpoint completion:

```bash
sbatch scripts/hpc/run_rl_grpo_phi4_smoke.slurm
```

Full RL after smoke passes:

```bash
sbatch scripts/hpc/run_rl_grpo_phi4_full.slurm
```

## Known Limitations

- Rewards are deterministic and do not inspect images.
- MIRAGE/AgBase consultation rewards are format and text-verifier based; they are not LLM-judged.
- Management coverage is keyword overlap only and intentionally capped.
- Smoke/full RL remain blocked until the final SFT checkpoint or adapter is present.
- The current worktree still shows pre-existing SFT-related dirty files; they were not modified by this task.
