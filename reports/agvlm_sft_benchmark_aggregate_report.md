# AGVLM SFT and Benchmark Aggregate Report

Date: 2026-05-18

## Executive Decision

Do not promote the new balanced-v2 SFT final checkpoint, and do not use it as the starting point for RL/GRPO.

Keep the previous SFT checkpoint active:

`/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1`

The new SFT final checkpoint regresses on the main held-out task metrics versus both the previous SFT and the original Phi-4 reasoning vision model. A recovered intermediate checkpoint sweep found that the best new-round checkpoint is `ckpt-1250`, but it still does not beat the previous SFT on task macro.

## Actions Completed After This Report

These follow-up actions were executed on 2026-05-18.

| Action | Status | Artifact |
| --- | --- | --- |
| Keep previous SFT active in benchmark config | Done | `benchmarks/vlm_baselines/agvlm_checkpoint_models.yaml` now loads base `microsoft/Phi-4-reasoning-vision-15B` with adapter `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1` |
| Add hard SFT promotion gate | Done | `scripts/eval/check_sft_promotion_gate.py`; gate report: `reports/sft_regression_audit/new_sft_promotion_gate.md` |
| Apply gate to new SFT vs previous SFT | Done, rejected | New SFT failed task macro, VQA relaxed accuracy, clarify macro F1, and invalid-output count |
| Validate active previous SFT adapter | Done | Adapter has `320` tensors and `320` non-empty tensors; first tensor `base_model.model.model.layers.0.mlp.down_proj.lora_A.weight` shape `[256, 17920]` |
| Add checkpoint-save and downstream checkpoint validation | Done | SFT post-save writes `adapter_validation.json`; RL and benchmark validators reject empty or tensorless PEFT adapters |
| Add output-format reward penalty | Done | New reward module `output_format` penalizes empty `Answer:`, generic labels such as `plant disease`, missing clarify decisions, missing consultation sections, and runaway repetition |
| Enable output-format penalty in Phi-4 RL configs | Done | Updated Phi-4 readiness, smoke, full, Turin tiny smoke, and step-eval configs |
| Run reward sanity check after reward change | Done | `reports/sft_regression_audit/rl_reward_sanity_after_output_format.md`; 200 sampled rows, `0` assertion failures |
| Add classification alias diagnostic to benchmark metrics | Done | Benchmark metrics now report `classification_accepted_label_accuracy` and `classification_semantic_alias_accuracy` in addition to exact label metrics |
| Submit previous SFT to benchmark runner on Turin | Submitted, pending | Slurm job `32655370` with `QUANTIZATION=4bit`, pending reason `Priority`; output dir `benchmarks/vlm_baselines/results/agvlm_previous_sft_benchmark_20260518_adapter_4bit_runtimefix`, logs `benchmarks/vlm_baselines/logs/slurm/agri-sft-ckpt-bench-32655370.out/.err` |
| Start RL benchmark | Blocked intentionally | No completed RL checkpoint exists; the failed new SFT is not allowed as an RL starting point |

Verification:

- Syntax check passed for modified Python scripts and modules.
- Targeted tests passed: `tests/test_checkpointing.py`, `tests/test_reward_functions.py`, `tests/test_benchmark_checkpoint_config.py`, `tests/test_benchmark_metrics.py`, `tests/test_rl_readiness_pipeline.py`.
- Benchmark config validation passed under the project conda environment for `agvlm_phi4_sft_completed`.
- An initial benchmark submission (`32654981`) failed early because the PEFT adapter directory was configured as a merged full checkpoint. The config was corrected to use `adapter_path` and validated. A second pending submission (`32655243`) was canceled before start. Job `32655297` then exposed a cluster runtime-cache permission issue, so `run_sft_benchmark_24gb.sbatch` was patched to place `XDG_RUNTIME_DIR`, Triton cache, and TorchInductor cache under job `TMPDIR`; the benchmark was resubmitted as `32655370`.

Reward sanity highlights after adding `output_format`:

| Candidate Type | Average Reward |
| --- | ---: |
| target answer / known good | `1.5455` |
| structured consultation | `0.5443` |
| empty output | `-1.0000` |
| generic overconfident answer | `-2.8494` |

Promotion-gate result for the new SFT:

| Required Metric | Previous SFT | New SFT | Pass |
| --- | ---: | ---: | --- |
| Task macro average | `0.133731` | `0.066908` | no |
| VQA relaxed accuracy | `0.128000` | `0.020000` | no |
| Clarify macro F1 | `0.243243` | `0.166667` | no |
| Invalid predictions | `216` | `244` | no |

## Source Artifacts

| Area | Artifact |
| --- | --- |
| SFT prep | `reports/sft_retrain_prep/sft_retrain_prep_report.md` |
| SFT training graphs | `reports/sft_training_graphs/round1_full/summary.md`, `reports/sft_training_graphs/round2_current_early/summary.md` |
| Data audit | `reports/sft_regression_audit/sft_train_phi4_max3_no_eval_overlap_target_quality.md`, `reports/sft_regression_audit/sft_train_phi4_max3_balanced_v2_instructional_labelrepaired_target_quality.md` |
| Full SFT comparison | `outputs/inference_checks/sft-round-comparison-phi4-512-turin-20260518T141832Z/` |
| New SFT checkpoint sweep | `outputs/inference_checks/sft-checkpoint-sweep-recovered-new-round-128-turin-20260518T163229Z/` |
| Previous SFT vs base gates | `outputs/inference_checks/sft-vs-base-phi4-512-turin16-greedy-min2-20260514T174318Z/`, `outputs/inference_checks/sft-vs-base-phi4-512-benchmarkprompt128-turin16-greedy-min2-20260514T183129Z/`, `outputs/inference_checks/sft-vs-base-benchmarks-rlholdout4096-promptaligned256-repairedimg-turin16-20260514T191952Z/` |
| External VLM benchmark report | `benchmarks/vlm_baselines/results/baseline_report_20260516/metrics/summary_table.md` |
| Benchmark split report | `benchmarks/vlm_baselines/splits/benchmark_split_report.md` |
| Benchmark smoke/debug runs | `benchmarks/vlm_baselines/results/phi4mm_*`, `benchmarks/vlm_baselines/results/metrics/summary_table.md` |

## SFT Stage Timeline

| Stage | Status | Main Findings |
| --- | --- | --- |
| SFT retrain prep | Completed | Built balanced-v2 and label-repaired manifests. No training was submitted during this prep step. |
| Round 1 SFT, full max3 | Completed artifact available | This is the previous active checkpoint. Training loss fell strongly and eval loss improved, but downstream gates were mixed and clarify behavior remained weak. |
| Round 2 balanced-v2 SFT | Training produced checkpoints, but Slurm state ended failed | Logs show training reached step 2813 and epoch 1.0. Final adapter files in checkpoint directories were empty, so adapters had to be recovered from ZeRO state. |
| New SFT full comparison | Completed on Turin/L4 | New final SFT regressed badly on task macro, invalid outputs, and VQA. |
| New SFT checkpoint sweep | Completed on Turin/L4 | `ckpt-1250` is the best new-round checkpoint, but it is still below previous SFT on task macro. Later checkpoints steadily degrade. |

## Data and Manifest Audit

| Manifest | Rows | Clarify | Classification | Consultation | VQA | Answer Median Tokens | Answer Mean Tokens | Numeric Label Prefix | Short Answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old SFT train | 292514 | 6482 | 86228 | 60114 | 139690 | 4.0 | 27.826 | 40624 / 13.8879% | 94230 / 32.2138% |
| New balanced train | 180000 | 27000 | 54000 | 36000 | 63000 | 6.0 | 35.306 | 10 / 0.0056% | 56739 / 31.5217% |

Findings:

- The new label-repaired manifest fixed the IP102 numeric label-prefix issue.
- The short-answer rate did not materially worsen.
- The task mix changed substantially: the new manifest increases clarify/respond proportionally and reduces total VQA/classification volume.
- Clarify/respond is repeated from only 6482 unique source examples into 27000 rows, so overfitting to narrow phrasing is a risk.

## SFT Training Metrics

| Run | Steps | Epoch | Loss First | Loss Last | Loss Min | Loss Mean | Eval Loss First | Eval Loss Last | Train Loss | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 1 full max3 | 4571 | 1.0 | 14.6309 | 2.9700 | 2.2408 | 3.5062 | 3.2955 | 2.6308 | 1.3352 | 194699.96s |
| Round 2 balanced-v2 | 2813 | 1.0 | 14.3479 | 3.5083 | 2.4142 | 3.9010 | 4.3732 | 4.4725 | 0.3718 | 32019.66s |

Training interpretation:

- Round 1 had a normal-looking optimization curve and improving eval loss.
- Round 2 train loss dropped, but eval loss was high and increased from `4.3732` to `4.4725`.
- Round 2 final outputs regressed despite lower train loss, which points to overfitting, target/prompt mismatch, reward mismatch, or output-format collapse rather than simple under-training.

## Previous SFT vs Base Gates

These were earlier inference gates on the previous SFT checkpoint.

| Run | Examples | Generation | Main Metrics | Interpretation |
| --- | ---: | --- | --- | --- |
| Max8 smoke | 8 | local holdout | Base avg reward `-0.021875`, SFT `0.000000` | Smoke only. SFT loaded and produced outputs. |
| 512 stratified, first full check | 512 | 16 Turin GPUs | Base avg reward `0.018359`, SFT `0.035034`; prediction changes `512/512` | SFT was non-empty and different from base, but local reward only. |
| 512 stratified, greedy min2 | 512 | `min_new_tokens=2`, `max_new_tokens=96` | Base avg reward `0.012036`, SFT `0.036792`; answer EM `0.001953` vs `0.005859` | SFT improved local reward and exact match slightly, but clarify remained broken. |
| 512 benchmark prompt, max128 | 512 | benchmark-aligned prompt | Base avg reward `0.261694`, SFT `0.244141`; both predicted clarify for all clarify cases | Prompt alignment exposed weaker SFT reward than base and clarify over-triggering. |
| 4096 prompt-aligned RL holdout | 4096 | `max_new_tokens=256` | Base avg reward `0.208286`, SFT `0.114060`; SFT answer EM `0.001221`; both had `premature_answer_rate=1.0` | Strong warning: previous SFT did not clear a robust RL-start gate on this holdout. |

Interpretation:

- The previous SFT is the best available trained checkpoint, but it was never a strong promotion-quality SFT by the benchmark-aligned gates.
- It remains better than the new final SFT for the current model line because the new round regressed further.

## Full 512-Example SFT Comparison

Run: `outputs/inference_checks/sft-round-comparison-phi4-512-turin-20260518T141832Z/`

Manifest: `data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl`

Generation: `greedy, min_new_tokens=2, max_new_tokens=256, 4-bit Turin/L4 inference`

| Model | Examples | Invalid | Empty | Task Macro | Class Top1 | Class F1 | VQA Relaxed | Clarify F1 | Local Avg Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phi4 Base | 512 | 232 | 0 | 0.161315 | 0.008547 | 0.008701 | 0.232000 | 0.243243 | -0.137459 |
| Previous SFT | 512 | 216 | 0 | 0.133731 | 0.029915 | 0.029948 | 0.128000 | 0.243243 | -0.030526 |
| New SFT final | 512 | 244 | 0 | 0.066908 | 0.008547 | 0.014056 | 0.020000 | 0.166667 | 0.017952 |

Findings:

- New SFT final has the worst task macro: `0.066908`.
- New SFT final has the worst invalid output count: `244`.
- New SFT final collapses on VQA relaxed accuracy: `0.020000`.
- The local average reward ranks new SFT highest, but task metrics rank it last. The local reward is not promotion-safe.

## New SFT Recovered Checkpoint Sweep

The original new-round checkpoint directories contained empty PEFT adapter files, so the LoRA adapters were recovered from DeepSpeed ZeRO checkpoint state.

Recovered adapter root:

`outputs/inference_checks/recovered-new-sft-zero-adapters-20260518T154604Z/`

Recovery validation:

- `num_tensors`: 320
- first tensor: `base_model.model.model.layers.0.mlp.down_proj.lora_A.weight`
- tensor shape: `[256, 17920]`
- dtype: `torch.bfloat16`

Sweep run: `outputs/inference_checks/sft-checkpoint-sweep-recovered-new-round-128-turin-20260518T163229Z/`

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

Findings:

- `ckpt-1250` is the best new-round checkpoint by task macro, but it still loses to previous SFT.
- `ckpt-1250` and `ckpt-1500` improve classification on this small slice, but they lose on VQA and have more invalid predictions.
- The trend is monotonic degradation after early checkpoints. The new round trains in the wrong direction for the current eval distribution.

## Benchmark Splits

| Phase | Val Rows | Test Rows | Duplicate IDs | Missing Images | Sample-ID Overlap | Group Overlap | Public Test Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rl_benchmark` | 369 | 1573 | 0 | 0 | 0 | 0 | 0 |
| `sft_benchmark` | 120 | 392 | 0 | 0 | 0 | 0 | 0 |

Task mix:

| Phase | Source/Task Summary |
| --- | --- |
| `sft_benchmark` | Test run metrics use 392 examples: classification, VQA, and clarify/respond. |
| `rl_benchmark` | Test run metrics use 1573 examples and include classification, VQA, clarify/respond, and consultation. |

## External VLM Benchmark Metrics

Run: `benchmarks/vlm_baselines/results/baseline_report_20260516/`

Generation: deterministic, `temperature=0.0`, `num_beams=1`; max tokens are benchmark dependent.

### SFT Benchmark Test Split

| Model | Examples | Task Macro | Class F1 | VQA Relaxed | Clarify F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA OneVision Qwen2 7B | 392 | 0.347833 | 0.019126 | 0.320000 | 0.704374 |
| Molmo2 4B | 392 | 0.326311 | 0.013115 | 0.284000 | 0.681818 |
| Qwen2.5-VL 3B | 392 | 0.266258 | 0.040710 | 0.300000 | 0.458065 |
| SmolVLM2 2.2B | 392 | 0.206707 | 0.004372 | 0.272000 | 0.343750 |
| Phi-4 Multimodal Instruct | 392 | 0.187670 | 0.019672 | 0.292000 | 0.251337 |
| PaliGemma2 3B | 392 | 0.170414 | 0.000000 | 0.268000 | 0.243243 |

### RL Benchmark Test Split

| Model | Examples | Task Macro | Class F1 | VQA Relaxed | Clarify F1 | Consultation Structured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phi-4 Multimodal Instruct | 1573 | 0.446985 | 0.010228 | 0.258278 | 0.565657 | 0.953778 |
| LLaVA OneVision Qwen2 7B | 1573 | 0.429843 | 0.011361 | 0.350993 | 0.437908 | 0.919111 |
| Molmo2 4B | 1573 | 0.425458 | 0.011409 | 0.245033 | 0.475610 | 0.969778 |
| Qwen2.5-VL 3B | 1573 | 0.210210 | 0.030875 | 0.258278 | 0.437908 | 0.113778 |
| SmolVLM2 2.2B | 1573 | 0.143451 | 0.010060 | 0.271523 | 0.283333 | 0.008889 |
| PaliGemma2 3B | 1573 | 0.092692 | 0.001820 | 0.324503 | 0.044444 | 0.000000 |

### Other Benchmark Artifacts Inspected

These runs were useful for debugging the benchmark runner or Phi-4 multimodal launch settings, but they are not decision-grade because they are tiny, partial, or have missing summary fields.

| Artifact | Examples | Reported Task Macro | Use in This Report |
| --- | ---: | ---: | --- |
| `benchmarks/vlm_baselines/results/phi4mm_rl_hd4_smoke_20260516/` | 120 | 0.311751 | Smoke/debug only. |
| `benchmarks/vlm_baselines/results/phi4mm_rl_bf16_128_smoke_20260516/` | 4 | 0.400000 | Smoke/debug only. |
| `benchmarks/vlm_baselines/results/phi4mm_rl_bf16_64_smoke_20260516/` | 120 |  | Smoke run with no task macro. |
| `benchmarks/vlm_baselines/results/phi4mm_smoke_20260516b/` | 2 |  | Smoke run with no task macro. |
| `benchmarks/vlm_baselines/results/phi4mm_smoke_20260516c/` | 2 |  | Smoke run with no task macro. |
| `benchmarks/vlm_baselines/results/phi4mm_smoke_20260516d/` | 2 | 0.000000 | Smoke/debug only. |
| `benchmarks/vlm_baselines/results/metrics/summary_table.md` | 512 per row | missing model metadata | Earlier baseline table; superseded by `baseline_report_20260516`. |

Benchmark interpretation:

- The strongest external baselines are much higher than the current AGVLM SFT gates on task macro.
- Classification macro F1 is low across all external baselines, largely because exact agricultural label-space matching is hard and many outputs are semantically plausible but outside the accepted label string.
- Consultation structured-section compliance can be high even when content is questionable. It should not be treated as factual correctness by itself.
- Direct comparison between external baseline tables and the 512 SFT regression check is not exact because the manifests differ. Still, they show the model needs stronger format adherence and task-specific calibration before RL.

## Inference Examples

### Full 512 SFT Comparison Examples

| Dataset | Task | Reference | Phi4 Base | Previous SFT | New SFT final |
| --- | --- | --- | --- | --- | --- |
| `ip102` | classification | `23 corn borer` | Refuses to identify the pest. | "The image shows a caterpillar..." | `aphid` |
| `plantdoc` | classification | `tomato mold leaf` | Refuses with unrelated safety text. | Describes yellowing leaves and possible fungal issue. | `Answer:` |
| `plantvillage_vqa` | VQA | `No` | Says yes, potato leaf with Septoria. | Correctly says it is not potato with Septoria. | `plant disease` |
| `plantvillage_vqa` | VQA | Huanglongbing cause | "A single green leaf..." | "small, round, and smooth seedling..." | `plant.` |
| `mirage` | clarify/respond | ask a clarifying question | Gives direct treatment advice. | Gives direct treatment advice. | `Answer:` |

### Checkpoint Sweep Examples

| Dataset | Task | Reference | Previous SFT | Best New ckpt-1250 | Later New ckpts |
| --- | --- | --- | --- | --- | --- |
| `ip102` | classification | `23 corn borer` | Caterpillar-like pest | `aphid` | mostly `aphid` |
| `plantdoc` | classification | `tomato mold leaf` | Yellowing leaves, possible nutrient or fungal issue | `bacterial spot` | mostly `Answer:` |
| `plantvillage_vqa` | VQA | `No` | Correctly says not potato Septoria | `: disease` | mostly `plant disease` |
| `mirage` | clarify/respond | ask clarification | Direct early-blight advice | `Answer:` | mostly `Answer:` |

### Benchmark Examples

| Source | Example | Observation |
| --- | --- | --- |
| SFT benchmark external baseline | LLaVA predicts `Answer: Caterpillar` for reference `23 corn borer`. | Semantically near, but invalid under exact label evaluation. |
| RL benchmark external baseline | Phi-4 Multimodal gives all required consultation sections for a live oak bark issue. | Strong structured compliance, but content is still image-only and needs factual validation. |
| Prompt-aligned 4096 holdout | For a tomato spider-mite yes/no item, base answers `Answer: Yes`; previous SFT outputs `to the`. | Previous SFT still has brittle generation under some prompts. |
| Prompt-aligned 4096 holdout | For live oak bark symptoms, both base and previous SFT produce structured consultation answers. | Structure exists, but exact diagnosis and management quality need stricter evaluation. |

## Cross-Stage Analysis

1. The new SFT is not a promotion candidate.

The full 512-example comparison is decisive: new SFT final is lower on task macro, VQA relaxed accuracy, clarify F1, and invalid count. The checkpoint sweep shows this is not just a bad final adapter; the whole new-round trajectory degrades after early checkpoints.

2. Round 2 likely overfit or learned an unstable output format.

Round 2 train loss dropped sharply, but eval loss was high and slightly worse at the second eval point. The generated outputs often collapse to short generic strings such as `plant.`, `aphid`, `Answer:`, `::`, or incomplete prefixes.

3. Data repair helped one known issue but did not fix the model behavior.

The numeric label-prefix problem was fixed, and short-answer rate did not worsen. The remaining problem is more likely a combination of changed task mix, repeated clarify data, prompt/target mismatch, and insufficient promotion gates during training.

4. The local reward is not aligned enough to drive model promotion.

The new SFT has the best local average reward in the 512 comparison, but it is clearly worse on task metrics and examples. Promotion needs to be gated by benchmark-style task metrics, not average reward alone.

5. Previous SFT is the least-bad trained checkpoint, not a final-quality model.

Previous SFT beats new SFT and improves classification versus base in the full comparison, but it still loses to base on VQA and task macro in the 512 comparison. Earlier benchmark-aligned gates also showed weak clarify behavior and low exact answers.

6. External baselines reveal the target bar.

External models reach task macro around `0.33-0.45` depending on split. The current AGVLM SFT checks are below that, especially for VQA and task robustness. Exact classification labels remain hard for all models, so better synonym/label evaluation is also needed.

## Next Steps

### Immediate

1. Keep previous SFT active in downstream configs.

Use:

`/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1`

Do not use:

`outputs/inference_checks/recovered-new-sft-zero-adapters-20260518T154604Z/checkpoint-2813`

2. Do not start RL/GRPO from the new SFT.

The model should not be optimized further from a checkpoint that already collapsed on VQA and output format.

3. Treat `ckpt-1250` only as a diagnostic artifact.

It is the best recovered new-round checkpoint, but it does not beat previous SFT on task macro and has more invalid outputs.

### Before Another SFT Run

1. Add a hard promotion gate during SFT training.

Evaluate every 100-250 steps on a fixed held-out set. Stop if task macro, VQA relaxed accuracy, or invalid output rate worsens for consecutive checkpoints.

Minimum gate:

- New checkpoint beats previous SFT on task macro.
- New checkpoint does not regress VQA relaxed accuracy.
- Invalid output count is lower than previous SFT.
- Clarify/respond macro F1 improves without predicting one decision for every case.
- Representative examples do not show `Answer:`, `plant.`, `::`, or other malformed outputs.

2. Fix checkpoint-save validation.

Add a post-save check that PEFT adapters contain non-empty LoRA tensors. If ZeRO recovery is required, automate it before inference comparison.

3. Rebalance the training manifest more conservatively.

Avoid repeating clarify/respond examples too aggressively. Keep enough VQA and classification coverage to prevent short-answer and generic-label collapse.

4. Add output-format hard negatives.

The model should be explicitly penalized or filtered for incomplete outputs such as `Answer:`, malformed prefixes, and generic labels when a specific label is required.

5. Revisit the reward modules.

The local reward preferred the regressed new SFT. Before RL, adjust reward weighting or add penalties for invalid/malformed outputs, VQA collapse, and missing required decision/section fields.

### Benchmark Work

1. Run the previous SFT and any future candidate through the same `benchmarks/vlm_baselines` evaluator used for external baselines.

Use both:

- `benchmarks/vlm_baselines/splits/sft_test_manifest.jsonl`
- `benchmarks/vlm_baselines/splits/rl_test_manifest.jsonl`

2. Report candidate checkpoints beside the external baselines.

The current external baseline table is useful, but the AGVLM checkpoints need to be evaluated with the same benchmark runner for direct apples-to-apples comparison.

3. Add semantic label aliases for benchmark classification.

Several outputs are near-correct but invalid by exact label string. Keep exact metrics, but also report semantic alias accuracy for pest/disease labels.

4. Separate consultation structure from consultation correctness.

Structured-section compliance is useful, but it should be paired with evidence quality, uncertainty quality, management keyword coverage, and forbidden-claim checks.

## Final Recommendation

Freeze the previous SFT as the active checkpoint, archive the new SFT final as a failed run, and start the next SFT attempt from the last known-good recipe with tighter validation. The next run should be short, checkpointed frequently, and stopped as soon as held-out task macro or VQA begins to degrade.
