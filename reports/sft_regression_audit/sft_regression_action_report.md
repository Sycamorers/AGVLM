# SFT Regression Action Report

Date: 2026-05-18

## Decision

Do not promote the new SFT final checkpoint, and do not use it as the starting point for RL/GRPO. Keep the previous SFT checkpoint active:

`/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1`

The new SFT regresses on the main task metrics against both the previous SFT and the original Phi-4 model. The recovered intermediate checkpoint sweep also does not find a new-round checkpoint that beats the previous SFT on task macro.

## Runs

- Canceled the B200 request: Slurm job `32636071`.
- Ran the three-way comparison on Turin/L4: Slurm job `32639163`.
- Ran the recovered checkpoint sweep on Turin/L4: Slurm job `32649038`.
- Generation for the full comparison: `greedy, min_new_tokens=2, max_new_tokens=256, 4-bit Turin/L4 inference`.
- Full comparison output: `outputs/inference_checks/sft-round-comparison-phi4-512-turin-20260518T141832Z/`.
- Checkpoint sweep output: `outputs/inference_checks/sft-checkpoint-sweep-recovered-new-round-128-turin-20260518T163229Z/`.

## Full 512-Example Comparison

Manifest: `data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl`

| Model | Examples | Invalid | Task Macro | Class Top1 | Class F1 | VQA Relaxed | Clarify F1 | Local Avg Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phi4 Base | 512 | 232 | 0.161315 | 0.008547 | 0.008701 | 0.232000 | 0.243243 | -0.137459 |
| Previous SFT | 512 | 216 | 0.133731 | 0.029915 | 0.029948 | 0.128000 | 0.243243 | -0.030526 |
| New SFT final | 512 | 244 | 0.066908 | 0.008547 | 0.014056 | 0.020000 | 0.166667 | 0.017952 |

Interpretation:

- New SFT final is a clear regression on task macro: `0.066908`, versus `0.133731` for previous SFT and `0.161315` for Phi4 Base.
- New SFT final has the worst invalid count among the three: `244`.
- New SFT final collapses especially hard on VQA relaxed accuracy: `0.020000`, versus `0.128000` for previous SFT and `0.232000` for Phi4 Base.
- The local average reward is higher for the new SFT, but that reward is not aligned with the benchmark-style task metrics in this run, so it should not drive promotion.

## Inference Examples

These examples are from the full 512-example report at `outputs/inference_checks/sft-round-comparison-phi4-512-turin-20260518T141832Z/pairwise_comparison.md`.

| Dataset | Task | Reference | Phi4 Base | Previous SFT | New SFT final |
| --- | --- | --- | --- | --- | --- |
| `ip102` | classification | `23 corn borer` | Refuses to identify the pest. | "The image shows a caterpillar..." | `aphid` |
| `plantdoc` | classification | `tomato mold leaf` | Refuses with unrelated safety text. | Describes yellowing leaves and possible fungal issue. | `Answer:` |
| `plantvillage_vqa` | vqa | `No` | Says yes, potato leaf with Septoria. | Correctly says it is not potato with Septoria. | `plant disease` |
| `plantvillage_vqa` | vqa | Huanglongbing cause | "A single green leaf..." | "small, round, and smooth seedling..." | `plant.` |
| `mirage` | clarify_or_respond | ask a clarifying question | Gives direct treatment advice. | Gives direct treatment advice. | `Answer:` |

The qualitative pattern matches the metrics: the new SFT often emits short generic labels such as `plant.`, `aphid`, `Answer:`, or malformed prefixes instead of useful agricultural answers.

## Checkpoint Sweep

The original new-round checkpoint directories had empty PEFT adapter files, so I recovered the LoRA adapters from the DeepSpeed ZeRO checkpoints before sweeping them.

Recovered adapter root:

`outputs/inference_checks/recovered-new-sft-zero-adapters-20260518T154604Z/`

Validation for the recovered adapters:

- `num_tensors`: 320
- first tensor: `base_model.model.model.layers.0.mlp.down_proj.lora_A.weight`
- first tensor shape: `[256, 17920]`
- dtype: `torch.bfloat16`

Sweep settings:

- Selected examples: 128
- Generation: `greedy, min_new_tokens=2, max_new_tokens=128`
- Checkpoints: `1250,1500,1750,2000,2250,2500,2750,2813`

| Model | Examples | Invalid | Task Macro | Class Top1 | Class F1 | VQA Relaxed | Clarify F1 | Local Avg Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous SFT | 128 | 56 | 0.068589 | 0.033333 | 0.060606 | 0.145161 | 0.000000 | -0.033887 |
| New SFT ckpt-1250 | 128 | 73 | 0.057511 | 0.050000 | 0.075758 | 0.096774 | 0.000000 | 0.004004 |
| New SFT ckpt-1500 | 128 | 69 | 0.046758 | 0.050000 | 0.075758 | 0.064516 | 0.000000 | 0.006836 |
| New SFT ckpt-1750 | 128 | 64 | 0.025904 | 0.033333 | 0.045455 | 0.032258 | 0.000000 | 0.016406 |
| New SFT ckpt-2000 | 128 | 64 | 0.015803 | 0.016667 | 0.015152 | 0.032258 | 0.000000 | 0.003906 |
| New SFT ckpt-2250 | 128 | 61 | 0.015803 | 0.016667 | 0.015152 | 0.032258 | 0.000000 | 0.003906 |
| New SFT ckpt-2500 | 128 | 61 | 0.010427 | 0.016667 | 0.015152 | 0.016129 | 0.000000 | -0.003906 |
| New SFT ckpt-2750 | 128 | 61 | 0.005051 | 0.016667 | 0.015152 | 0.000000 | 0.000000 | -0.011719 |
| New SFT ckpt-2813 | 128 | 61 | 0.005051 | 0.016667 | 0.015152 | 0.000000 | 0.000000 | -0.011719 |

Checkpoint sweep conclusion:

- Best new-round checkpoint by task macro is `ckpt-1250`, but it is still below previous SFT: `0.057511` versus `0.068589`.
- `ckpt-1250` and `ckpt-1500` have slightly better classification metrics on this 128-example slice, but they lose on VQA and have more invalid predictions than previous SFT.
- Later checkpoints steadily degrade. This suggests the new SFT round continues training in the wrong direction for the current eval distribution.

## Data Audit

| Manifest | Rows | Clarify | Classification | Consultation | VQA | Answer Median Tokens | Answer Mean Tokens | Numeric Label Prefix | Short Answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old SFT train | 292514 | 6482 | 86228 | 60114 | 139690 | 4.0 | 27.826 | 40624 / 13.8879% | 94230 / 32.2138% |
| New balanced train | 180000 | 27000 | 54000 | 36000 | 63000 | 6.0 | 35.306 | 10 / 0.0056% | 56739 / 31.5217% |

Audit outputs:

- `reports/sft_regression_audit/sft_train_phi4_max3_no_eval_overlap_target_quality.md`
- `reports/sft_regression_audit/sft_train_phi4_max3_balanced_v2_instructional_labelrepaired_target_quality.md`

Interpretation:

- The numeric IP102 label-prefix issue is effectively fixed in the new balanced manifest.
- The short-answer rate did not materially worsen.
- The task mix changed substantially. The new manifest has much more `clarify_or_respond` proportionally and fewer total VQA/classification examples. That shift is a likely contributor, but the metrics also show output-format collapse during training, especially after `ckpt-1250`.

## Recommended Next Step

Keep the previous SFT checkpoint in all downstream configs. For the next SFT attempt, rerun from the previous known-good setup with a smaller validation cadence and an early-stop gate on task macro, invalid-output rate, and VQA relaxed accuracy. Do not continue from the new final checkpoint.
