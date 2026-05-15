# SFT Retrain Preparation Report

## Status

No SFT or RL training was submitted. The previous unprepared Slurm job was cancelled before full training, and this prep step only generated manifests, configs, and audits.

## Prepared Artifacts

- Balanced manifest config: `configs/data/sft_balanced_v2_phi4_max3.yaml`
- Label repair config: `configs/data/sft_label_repair_balanced_v2_phi4_max3.yaml`
- Prompt/target audit config: `configs/data/sft_format_audit_balanced_v2_phi4_max3.yaml`
- Pilot training config, not submitted: `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_balanced_v2_instructional_pilot.yaml`
- Balanced manifest: `data/manifests/full/sft_train_phi4_max3_balanced_v2_instructional.jsonl`
- Label-repaired balanced manifest: `data/manifests/full/sft_train_phi4_max3_balanced_v2_instructional_labelrepaired.jsonl`
- Format audit: `reports/sft_retrain_prep/sft_format_audit_balanced_v2_instructional_labelrepaired.md`

## Balanced Manifest

The new SFT manifest has 180,000 rows with explicit task targets:

| Task | Rows |
| --- | ---: |
| vqa | 63,000 |
| classification | 54,000 |
| consultation | 36,000 |
| clarify_or_respond | 27,000 |

Compared with the previous 292,514-row training manifest, this reduces VQA dominance and increases clarify/respond from 6,482 source examples to 27,000 training rows by controlled repetition.

## Label Repair

The audit showed IP102 classification targets such as `45 alfalfa weevil` and `102 Cicadellidae`. The label-repaired manifest strips leading numeric class ids from the training target while keeping the original numeric form as an accepted label alias.

Repair summary:

| Source | Repaired Rows | Numeric Prefixes Remaining |
| --- | ---: | ---: |
| ip102 | 25,439 | 0 |

Example repaired target:

```text
Answer: rice water weevil
```

Accepted labels keep both forms:

```text
rice water weevil
11 rice water weevil
```

## Format Audit

The rendered instructional audit sampled 20 unique examples per task, 80 total. It found 0 syntax failures for the intended prompt/target contracts:

- classification and VQA targets use `Answer: ...`
- clarify/respond targets use `Decision: clarify` or `Decision: respond`
- consultation targets use `Diagnosis:`, `Evidence:`, `Uncertainty:`, `Management:`, and `Follow-up:`

## Remaining Risks Before Training

- Some consultation targets are still short diagnosis labels, especially from AGBase. They are now wrapped in structured sections, but content quality still depends on source metadata.
- Clarify/respond is still repeated from only 6,482 unique examples. This should help the decision boundary, but it may overfit repeated phrasing.
- The pilot should be treated as a gate, not as the final SFT. The model should be compared against base and benchmarks on the 4096 prompt-aligned holdout before any RL run.

## Pilot Gate

The prepared pilot config saves checkpoints every 100 steps and the final adapter:

```text
configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_balanced_v2_instructional_pilot.yaml
```

Minimum go/no-go criteria before RL:

- SFT task macro clearly beats base.
- VQA does not repeat the previous large regression.
- Clarify/respond F1 improves materially.
- Classification improves with semantic labels.
- Consultation remains structured and actual responses look usable in the pairwise report.
