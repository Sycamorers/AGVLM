# First SFT Round Analysis Report

## Executive Summary

The first Phi-4 reasoning vision SFT run completed technically and produced usable LoRA adapter weights, but it did not pass the quality gate for RL. The adapter improved output structure, especially consultation section formatting, but it regressed VQA sharply, did not learn clarify/respond decision behavior, and only weakly improved classification top-1 accuracy.

The correct decision is not to start RL from this first SFT checkpoint. We are now running a repaired second SFT attempt with prompt-aligned targets, a balanced task manifest, repaired IP102 labels, and checkpoint saving so intermediate models can be evaluated before RL.

## First-Round Training Run

| Item | Value |
| --- | --- |
| Model | `microsoft/Phi-4-reasoning-vision-15B` |
| SFT output | `outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1` |
| Saved adapter | `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1/adapter_model.safetensors` |
| Final checkpoint | `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1/checkpoint-4571` |
| Adapter size | about `1.7G` |
| Train rows | `292,514` |
| Train steps | `4,571` |
| Epochs | `1.0` |
| Final train loss | `1.3352` |
| Runtime | `194,699.96s`, about `54.1h` |
| Train throughput | `1.502` samples/sec |

The weights were saved correctly. This is a training-success result, but not a model-quality-success result.

## Training and Eval Data

The first round used this no-overlap training manifest:

`data/manifests/full/sft_train_phi4_max3_no_eval_overlap.jsonl`

| Task | Train Rows | Share |
| --- | ---: | ---: |
| VQA | 139,690 | 47.8% |
| classification | 86,228 | 29.5% |
| consultation | 60,114 | 20.6% |
| clarify/respond | 6,482 | 2.2% |

The training-time SFT eval was:

`data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl`

| Task | Eval Rows |
| --- | ---: |
| classification | 234 |
| VQA | 250 |
| clarify/respond | 28 |
| consultation | 0 |

This 512-row eval is acceptable as a cheap loss monitor, but it was not strong enough as an SFT acceptance gate. It excluded consultation entirely and had too few clarify/respond examples to judge decision behavior.

## First Diagnostic: 512 Prompt-Aligned Comparison

Report:

`outputs/inference_checks/sft-vs-base-benchmarks-phi4-512-benchmarkprompt128-diagnostic-20260514T184042Z/multi_model_pairwise_comparison.md`

| Metric | Phi4 Base | Phi4 SFT |
| --- | ---: | ---: |
| task macro average | 0.261232 | 0.243418 |
| local composite reward | 0.261694 | 0.244141 |
| classification top1 | 0.021368 | 0.051282 |
| classification macro F1 | 0.019441 | 0.025998 |
| VQA relaxed accuracy | 0.360000 | 0.300000 |
| VQA token F1 | 0.409685 | 0.319620 |
| clarify macro F1 | 0.404255 | 0.404255 |
| empty outputs | 0 | 0 |

Readout:

- The SFT improved classification top1 on this small gate, but the absolute score was still very low.
- The SFT regressed VQA.
- The SFT was below base on task macro and composite reward.
- Clarify/respond did not show a meaningful SFT improvement. On this 512 prompt, base and SFT had the same clarify metrics.

This was the first sign that the SFT was not robustly better than base.

## Main Gate: 4096 Prompt-Aligned RL Holdout

Report:

`outputs/inference_checks/sft-vs-base-benchmarks-rlholdout4096-promptaligned256-repairedimg-turin16-20260514T191952Z/multi_model_pairwise_comparison.md`

Metrics:

`outputs/inference_checks/sft-vs-base-benchmarks-rlholdout4096-promptaligned256-repairedimg-turin16-20260514T191952Z/multi_model_metrics.json`

Manifest split:

| Task | Rows |
| --- | ---: |
| VQA | 2,299 |
| classification | 1,498 |
| consultation | 256 |
| clarify/respond | 43 |

Metric-family counts can differ slightly from raw task counts because the evaluator groups some examples by verifier family.

## Model Comparison on 4096 Gate

| Model | Task Macro | Class Top1 | Class F1 | VQA Relaxed | VQA Token F1 | Clarify F1 | Consultation Sections | Invalid | Empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phi4 Base | 0.1880 | 0.0120 | 0.0144 | 0.2979 | 0.3055 | 0.0000 | 0.3953 | 1182 | 0 |
| Phi4 SFT | 0.2781 | 0.0347 | 0.0137 | 0.1114 | 0.0867 | 0.0000 | 0.9430 | 1170 | 0 |
| Molmo2-4B | 0.4236 | 0.0187 | 0.0094 | 0.2422 | 0.2774 | 0.9512 | 0.9672 | 1248 | 0 |
| LLaVA-OneVision-7B | 0.4058 | 0.0227 | 0.0105 | 0.2576 | 0.2991 | 0.7647 | 0.9172 | 1183 | 0 |
| Qwen2.5-VL-3B | 0.2063 | 0.0387 | 0.0312 | 0.2559 | 0.2586 | 0.7647 | 0.1000 | 1016 | 0 |
| SmolVLM2-2.2B | 0.1250 | 0.0234 | 0.0086 | 0.2002 | 0.2240 | 0.5000 | 0.0078 | 1266 | 0 |
| PaliGemma2-3B | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 4096 | 4096 |
| Phi-4-multimodal-instruct | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 4096 | 4096 |

Unavailable benchmark notes:

- `google/paligemma2-3b-mix-448` failed because the model repo was gated and unavailable to the job.
- `microsoft/Phi-4-multimodal-instruct` failed in this environment due to remote-code generation support incompatibility.

## Detailed Phi4 Base vs SFT Findings

### What Improved

The SFT strongly improved consultation structure:

| Metric | Base | SFT |
| --- | ---: | ---: |
| required section compliance | 0.3953 | 0.9430 |
| management keyword coverage | 0.2344 | 0.5625 |
| follow-up question presence | 0.3711 | 0.9258 |

It also slightly improved classification top1:

| Metric | Base | SFT |
| --- | ---: | ---: |
| classification top1 | 0.0120 | 0.0347 |
| invalid classification output rate | 0.7891 | 0.7677 |

However, these improvements are not enough to justify RL.

### What Regressed

VQA regressed badly:

| Metric | Base | SFT |
| --- | ---: | ---: |
| VQA relaxed accuracy | 0.2979 | 0.1114 |
| VQA token F1 | 0.3055 | 0.0867 |
| VQA exact match | 0.2550 | 0.0596 |
| VQA missing answer rate | 0.0000 | 0.0086 |

This is the clearest quality failure. A usable SFT cannot lose this much basic VQA ability before RL.

Clarify/respond was not learned:

| Metric | Base | SFT |
| --- | ---: | ---: |
| clarify F1 | 0.0000 | 0.0000 |
| clarify macro F1 | 0.0444 | 0.0444 |
| decision accuracy | 0.0465 | 0.0465 |
| under-clarification rate | 1.0000 | 1.0000 |

On the 4096 gate, both Phi4 base and first SFT failed to choose `clarify` for expected-clarify cases. This confirms the first SFT did not learn the decision boundary.

Consultation structure improved, but safety/content signals worsened in places:

| Metric | Base | SFT |
| --- | ---: | ---: |
| forbidden claim rate | 0.0039 | 0.0156 |
| repetition rate | 0.1114 | 0.3294 |

The SFT learned the headers, but that does not mean it learned better consultation content.

## Why the First SFT Was Not Good Enough

### 1. Prompt and Target Formatting Were Misaligned

The first SFT was trained mostly with the existing manifest prompt and plain target text. The later inference gates expected explicit contracts:

- `Answer: ...`
- `Decision: clarify`
- `Clarifying question: ...`
- `Decision: respond`
- `Diagnosis:`, `Evidence:`, `Uncertainty:`, `Management:`, `Follow-up:`

The model was being judged on a format it was not consistently trained to produce. This explains part of the invalid-output and parsing instability.

### 2. Clarify/Respond Was Too Small

Clarify/respond was only `6,482` rows out of `292,514`, about `2.2%` of training. That is too small for a behavior that matters heavily before RL. The first SFT did not learn when to ask a clarifying question versus when to answer.

### 3. The Evaluation Gate Was Too Weak

The 512-row SFT eval had no consultation rows and only 28 clarify/respond rows. It could monitor training loss, but it could not certify that the model was ready for RL.

The 4096 prompt-aligned gate was much more informative and exposed the real problems.

### 4. Classification Labels Had Source-Specific Noise

The IP102 labels included numeric prefixes such as `45 alfalfa weevil` and `102 Cicadellidae`. These are source class ids, not natural agricultural labels. They made the target format less natural and made exact matching harsher.

This has now been repaired for the next SFT manifest: the training target is the semantic label, while the original numeric form remains an accepted alias.

### 5. The SFT Learned Format More Than Semantics

The strongest gain was consultation section compliance. The strongest loss was VQA. That pattern suggests the first SFT pushed the model toward output templates without preserving enough task correctness.

For RL, this is risky. RL would optimize on top of a model that already has degraded visual QA ability and weak decision calibration.

## RL Readiness Decision

The first SFT should not be used as the RL starting point.

Minimum conditions before RL should be:

- SFT task macro beats base on the 4096 prompt-aligned gate.
- VQA does not materially regress versus base.
- Clarify/respond F1 improves meaningfully.
- Classification improves with semantic labels, not just parser artifacts.
- Consultation remains structured, but actual responses are manually inspected.
- Empty, missing, invalid, and repetitive outputs do not increase.

The first SFT fails these conditions because VQA regressed and clarify/respond did not improve.

## What We Are Doing Now

An unprepared retraining attempt was cancelled before full training:

- Cancelled job: `32417271`
- It only reached preflight.
- No full checkpoint directory was created for that cancelled attempt.

Then we prepared a corrected second SFT round.

### Data Repair

New balanced manifest:

`data/manifests/full/sft_train_phi4_max3_balanced_v2_instructional_labelrepaired.jsonl`

| Task | Rows |
| --- | ---: |
| VQA | 63,000 |
| classification | 54,000 |
| consultation | 36,000 |
| clarify/respond | 27,000 |
| total | 180,000 |

This changes the training mix from the first round by reducing VQA dominance and increasing clarify/respond exposure.

### Label Repair

IP102 repaired rows:

| Source | Repaired Rows | Numeric Prefixes Remaining |
| --- | ---: | ---: |
| ip102 | 25,439 | 0 |

Example target style after repair:

```text
Answer: rice water weevil
```

The original class-id form remains in accepted labels for evaluation compatibility.

### Prompt/Target Repair

The second SFT uses explicit instructional rendering:

- `sft_prompt_format: instructional`
- `sft_target_format: instructional`

Format audit:

`reports/sft_retrain_prep/sft_format_audit_balanced_v2_instructional_labelrepaired.md`

The audit sampled 80 rendered examples, 20 per task, and found `0` format validation failures.

### Tests

Focused tests passed:

```bash
PYTHONPATH=src pytest -q tests/test_manifest_builders.py tests/test_data_transforms.py tests/test_collators.py tests/test_sft_trainer.py tests/test_evaluation_pipeline.py tests/test_benchmark_phase_splits.py tests/test_benchmark_metrics.py tests/test_benchmark_prediction_parsing.py
```

Result: `46` tests passed.

### Current Full Training Job

Current job:

| Item | Value |
| --- | --- |
| Slurm job | `32418919` |
| State at last check | `RUNNING` |
| Partition | `hpg-turin` |
| Nodes | `8` |
| GPUs | `16` L4 GPUs total |
| CPUs | `64` |
| Config | `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_balanced_v2_instructional_full.yaml` |
| Checkpoint directory | `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1` |
| Save policy | every `250` steps |
| Final save | enabled |

At the last log check, the job was running the 16-rank preflight stage. The Slurm script will proceed to full training after the two-step preflight passes.

## Expected Difference Between First and Second SFT

| Area | First SFT | Current Second SFT |
| --- | --- | --- |
| Prompt format | mostly manifest/native | explicit instructional contracts |
| Target format | mostly plain targets | explicit `Answer:`, `Decision:`, consultation sections |
| Train rows | 292,514 | 180,000 |
| Clarify/respond rows | 6,482 | 27,000, repeated from 6,482 unique rows |
| IP102 labels | numeric prefixes in targets | semantic target labels, numeric aliases preserved |
| Eval decision | 512 eval was too weak | 4096 prompt-aligned gate required before RL |
| Checkpointing | final plus late checkpoints | save every 250 steps, keep more checkpoints |
| LR | `5e-6` | `3e-6` |

## Next Steps After This Training Finishes

1. Confirm the final adapter and checkpoints were saved.
2. Run inference on the saved checkpoints, not only the final model.
3. Compare base, first SFT, second SFT checkpoints, and benchmark models on the 4096 prompt-aligned gate.
4. Generate a pairwise Markdown report with metrics and actual responses.
5. Only start RL if the second SFT clears the go/no-go criteria above.

## Bottom Line

The first SFT was a useful diagnostic run, but not a good RL base. It showed that the training pipeline can finish and save weights, but it also exposed format mismatch, task imbalance, weak clarify behavior, noisy classification targets, and major VQA regression.

The current second SFT is designed to test whether those issues can be repaired with better data balance, explicit target formatting, label repair, lower learning rate, and checkpoint-level evaluation before RL.
