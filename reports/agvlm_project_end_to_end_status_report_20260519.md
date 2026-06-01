# AGVLM End-to-End Status Report

Date: 2026-05-19 UTC

Audience: readers who do not already know this project.

## 1. Executive Summary

AGVLM is an agriculture-focused vision-language model project. The model is intended to answer questions about ground-level RGB agricultural images, such as crop disease, pest identification, visual question answering, and whether the model should ask for more information before giving advice.

The most important current result is:

- The latest completed benchmark gate passed on benchmark job `32676558`.
- The automatic monitor submitted the next full SFT training job, `32677813`.
- Training job `32677813` launched on `hpg-turin` with 16 L4 GPUs but failed during the processor save step.
- We patched the SFT save path and resubmitted the same full training configuration as job `32679114`.
- Replacement job `32679114` completed successfully with exit code `0:0`.
- We benchmarked the completed candidate checkpoint as job `32680255`.
- We also benchmarked raw Phi-4 reasoning vision as job `32680256`.
- The completed candidate failed the promotion gate and should not replace the previous SFT checkpoint.

The gate passed because the benchmark completed with no runtime failures and met all configured quality thresholds:

| Metric | Actual | Required | Result |
| --- | ---: | ---: | --- |
| Benchmark examples | `392` | `>= 392` | pass |
| Runtime failure rate | `0.0` | `<= 0.0` | pass |
| Invalid prediction rate | `0.30357142857142855` | `<= 0.50` | pass |
| Task macro average | `0.22877419354838713` | `>= 0.13` | pass |
| VQA relaxed accuracy | `0.156` | `>= 0.12` | pass |
| Clarify/respond macro F1 | `0.5303225806451614` | `>= 0.20` | pass |

This does not mean the model is finished. It means the benchmark infrastructure is now working, the active previous SFT checkpoint was usable enough to clear the gate, the automated gate correctly launched full training, and the training save bug found in the first launched job has been fixed. The new completed SFT candidate improved VQA relaxed accuracy, but regressed task macro, clarify/respond F1, and invalid prediction count versus the previous SFT. Classification remains weak across raw Phi-4, previous SFT, and the new completed SFT: all three reported `classification_macro_f1 = 0.0`.

## 2. What the Current Step Does

The current step is an automatic benchmark-gated training launch followed by a training-save fix and resubmission.

In plain terms, it does this:

1. Submit a benchmark job for the active model checkpoint.
2. Watch that benchmark until it finishes.
3. If the benchmark crashes for infrastructure reasons, retry it.
4. If the benchmark finishes, read the metrics.
5. If the metrics pass the requirements, automatically submit the next full training job.
6. If the metrics fail, do not start training.
7. If the submitted training job fails for an engineering reason, debug the exact failure, patch it, and resubmit the same training configuration.
8. After a completed training job, validate the saved checkpoint artifacts and benchmark the candidate before promotion.

The active scripts are:

- Watcher: `scripts/hpc/watch_benchmark_until_success_then_launch_sft.slurm`
- Gate checker and training submitter: `scripts/hpc/benchmark_gate_then_submit_sft.py`
- Benchmark wrapper: `benchmarks/vlm_baselines/slurm/run_sft_benchmark_agvlm_checkpoint.sbatch`
- Full SFT training wrapper: `scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm`

Why we are doing this:

- Full training is expensive. We do not want to start it from a broken model, broken benchmark path, or failed inference configuration.
- Previous training rounds produced misleading local signals. Some runs had lower training loss or better local reward but worse real benchmark behavior.
- The gate forces a minimum benchmark standard before using cluster resources for another full training run.
- The watcher removes manual babysitting. It can retry infrastructure failures and only launch training when the benchmark is genuinely usable.

## 3. Project Background

The project is a config-driven research codebase for an agriculture-focused vision-language model. The current V1 scope is ground-level RGB agricultural consultation only. The model should not become a generic all-purpose VLM by default.

The base model is:

`microsoft/Phi-4-reasoning-vision-15B`

The main training method so far has been supervised fine-tuning, or SFT. In SFT, the model sees an image plus an instruction and is trained to produce a target answer.

The current active baseline SFT adapter is still:

`/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1`

The first full training job submitted by the gate was:

`32677813`

It failed after launch with:

`AttributeError: 'Phi4VisionRProcessor' object has no attribute 'chat_template'`

The replacement full training job was:

`32679114`

It completed successfully after the save path was patched.

The new candidate checkpoint is:

`/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1`

The new candidate was benchmarked and rejected by the promotion gate. The previous SFT checkpoint remains the active baseline.

## 4. Terms Used in This Report

| Term | Meaning |
| --- | --- |
| VLM | Vision-language model. A model that reads images and text and generates text. |
| SFT | Supervised fine-tuning. Training on examples where the desired answer is known. |
| RL / GRPO | Reinforcement-style training planned after SFT, using reward functions instead of only target answers. |
| Checkpoint | Saved model state during or after training. |
| Adapter | A small trainable LoRA/PEFT module stored separately from the base model. It modifies the base model without saving a full copy of all weights. |
| PEFT / LoRA | Parameter-efficient fine-tuning. This project stores trained changes as LoRA adapter tensors. |
| ZeRO checkpoint | DeepSpeed distributed checkpoint format. Sometimes adapter weights must be recovered from this format. |
| Benchmark | Held-out evaluation data not used for training. It measures whether the model actually performs the task. |
| Gate | Automatic pass/fail criteria. The model must clear the gate before training proceeds. |
| Invalid prediction | Output that is empty, malformed, not parseable, or not in the required answer format. |
| Task macro average | Average task score across task families so one large task does not dominate the result. |
| Slurm | Cluster job scheduler used to submit GPU jobs. |
| Turin | The cluster GPU partition used here, primarily L4 GPUs. |
| 4-bit inference | Quantized model loading to fit a large model on smaller GPUs. |

## 5. Main Data and Tasks

The project evaluates several agriculture task types:

| Task | What the model should do |
| --- | --- |
| Classification | Identify the crop issue, disease, pest, or label in the image. |
| VQA | Answer a specific visual question about the image. |
| Clarify/respond | Decide whether enough information exists to answer or whether the model should ask a clarifying question. |
| Consultation | Produce structured agricultural advice with sections such as diagnosis, evidence, management, and follow-up. |

Important held-out benchmark split:

`benchmarks/vlm_baselines/splits/sft_test_manifest.jsonl`

The latest successful benchmark used 392 SFT benchmark test examples:

- Classification: 114
- VQA: 250
- Clarify/respond: 28

## 6. Timeline of Work Completed So Far

### Stage 1: Built and Audited Training Data

We prepared SFT training and evaluation manifests. The old SFT training manifest had 292,514 rows. The new balanced-v2 manifest had 180,000 rows.

Key finding:

- The new data fixed the numeric label-prefix issue in IP102 labels.
- However, the new task mix changed substantially. Clarify/respond examples increased proportionally, and many were repeated from a smaller unique source pool.
- That raised the risk of overfitting to narrow clarify/respond patterns.

Relevant reports:

- `reports/sft_retrain_prep/sft_retrain_prep_report.md`
- `reports/sft_regression_audit/sft_train_phi4_max3_no_eval_overlap_target_quality.md`
- `reports/sft_regression_audit/sft_train_phi4_max3_balanced_v2_instructional_labelrepaired_target_quality.md`

### Stage 2: Round 1 SFT Completed

Round 1 is the previous active checkpoint.

Checkpoint:

`/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1`

Training summary:

| Run | Steps | Epoch | Loss first | Loss last | Eval loss first | Eval loss last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 1 full max3 | `4571` | `1.0` | `14.6309` | `2.9700` | `3.2955` | `2.6308` |

Interpretation:

- The optimization curve looked normal.
- Evaluation loss improved.
- Downstream behavior was still mixed, especially on VQA and clarify/respond.

### Stage 3: Round 2 Balanced-v2 SFT Regressed

Round 2 was intended to improve the model using the balanced-v2 data recipe.

Training summary:

| Run | Steps reached | Epoch | Loss first | Loss last | Eval loss first | Eval loss last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 2 balanced-v2 | `2813` | `1.0` | `14.3479` | `3.5083` | `4.3732` | `4.4725` |

What failed:

- The Slurm training state ended failed.
- Final adapter files in checkpoint directories were empty.
- The training loss fell, but evaluation loss was high and got worse.
- The final model output quality regressed.

How we recovered enough to evaluate it:

- We recovered PEFT adapter tensors from DeepSpeed ZeRO checkpoint state.
- The recovered adapter had 320 non-empty tensors.
- This allowed us to benchmark intermediate checkpoints even though normal adapter files were empty.

Relevant recovery artifact:

`outputs/inference_checks/recovered-new-sft-zero-adapters-20260518T154604Z/`

### Stage 4: Compared Original Phi, Previous SFT, and New SFT

We compared three models on a 512-example held-out SFT evaluation set:

- Original Phi-4 reasoning vision base model
- Previous SFT, also called Round 1
- New SFT final, also called Round 2 final

Run:

`outputs/inference_checks/sft-round-comparison-phi4-512-turin-20260518T141832Z/`

Results:

| Model | Examples | Invalid | Empty | Task macro | Classification top1 | Classification F1 | VQA relaxed | Clarify F1 | Local avg reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phi-4 base | 512 | 232 | 0 | 0.161315 | 0.008547 | 0.008701 | 0.232000 | 0.243243 | -0.137459 |
| Previous SFT | 512 | 216 | 0 | 0.133731 | 0.029915 | 0.029948 | 0.128000 | 0.243243 | -0.030526 |
| New SFT final | 512 | 244 | 0 | 0.066908 | 0.008547 | 0.014056 | 0.020000 | 0.166667 | 0.017952 |

Conclusion:

- The new SFT final checkpoint should not be promoted.
- It had the worst task macro average.
- It had the worst VQA relaxed accuracy.
- It had the highest invalid output count.
- The local reward was misleading because it ranked the new SFT highest even though task metrics were worse.

### Stage 5: Swept New SFT Intermediate Checkpoints

Because the final new SFT regressed, we checked whether an earlier checkpoint was better.

Run:

`outputs/inference_checks/sft-checkpoint-sweep-recovered-new-round-128-turin-20260518T163229Z/`

Results:

| Model | Examples | Invalid | Task macro | Classification top1 | Classification F1 | VQA relaxed | Clarify F1 | Local avg reward |
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

Conclusion:

- `ckpt-1250` was the best new-round checkpoint.
- It still did not beat previous SFT on task macro.
- Later checkpoints got steadily worse.
- This suggests the new training recipe moved in the wrong direction after early training.

### Stage 6: Added Safety Checks and Reward Fixes

We added several protections after seeing the regression:

- A hard SFT promotion gate.
- Checkpoint validation to reject empty or tensorless adapters.
- Benchmark config validation for SFT/RL checkpoints.
- Output-format reward penalties.
- Additional benchmark metrics for classification aliases.

New reward module:

`src/agri_vlm/rewards/output_format.py`

Why:

- The model often produced malformed outputs such as `Answer:`, `plant.`, `::`, or overly generic labels.
- The local reward previously did not punish these strongly enough.
- The new reward penalizes empty answers, generic labels, missing clarify decisions, missing consultation sections, and runaway repetition.

Reward sanity check result:

| Candidate type | Average reward |
| --- | ---: |
| target answer / known good | `1.5455` |
| structured consultation | `0.5443` |
| empty output | `-1.0000` |
| generic overconfident answer | `-2.8494` |

Sanity check artifact:

`reports/sft_regression_audit/rl_reward_sanity_after_output_format.md`

### Stage 7: Built External Benchmark Comparison

We compared external VLM baselines on the benchmark runner to understand the target bar.

Run:

`benchmarks/vlm_baselines/results/baseline_report_20260516/`

SFT benchmark test split:

| Model | Examples | Task macro | Classification F1 | VQA relaxed | Clarify F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA OneVision Qwen2 7B | 392 | 0.347833 | 0.019126 | 0.320000 | 0.704374 |
| Molmo2 4B | 392 | 0.326311 | 0.013115 | 0.284000 | 0.681818 |
| Qwen2.5-VL 3B | 392 | 0.266258 | 0.040710 | 0.300000 | 0.458065 |
| SmolVLM2 2.2B | 392 | 0.206707 | 0.004372 | 0.272000 | 0.343750 |
| Phi-4 Multimodal Instruct | 392 | 0.187670 | 0.019672 | 0.292000 | 0.251337 |
| PaliGemma2 3B | 392 | 0.170414 | 0.000000 | 0.268000 | 0.243243 |

Interpretation:

- External models still outperform the current AGVLM SFT line on many benchmark metrics.
- Classification is hard for all models under exact label matching.
- The current project still needs stronger format control and better task-specific calibration.

### Stage 8: Set Up Benchmark Watcher and Training Gate

We then created an automatic benchmark watcher:

`scripts/hpc/watch_benchmark_until_success_then_launch_sft.slurm`

This is the step the current user question is asking about.

It is not training by itself. It is a controller:

- It launches benchmark attempts.
- It checks whether benchmark attempts failed.
- It retries infrastructure failures.
- It reads benchmark metrics when a benchmark completes.
- It submits full SFT only if all gates pass.

The latest successful watcher was:

`32676538`

The successful benchmark attempt was:

`32676558`

The training job submitted by the gate was:

`32677813`

It failed during the processor save step, so the gate/submission mechanism succeeded but the first full training run did not exit cleanly.

We then patched `src/agri_vlm/training/sft_trainer.py` so SFT processor saving handles missing optional processor attributes before calling `save_pretrained(...)`. The same full training configuration was resubmitted as:

`32679114`

That replacement job completed successfully and produced the candidate checkpoint:

`/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1`

## 7. Failure History and Debugging Details

This section explains what failed, how it failed, what we tried, and why.

### Failure A: The New SFT Model Regressed

What failed:

- New SFT final performed worse than previous SFT and base Phi-4 on core task metrics.

How it failed:

- Task macro dropped to `0.066908`.
- VQA relaxed accuracy dropped to `0.020000`.
- Invalid outputs increased to `244 / 512`.
- Outputs often became generic, malformed, or incomplete.

What we tried:

- Compared base, previous SFT, and new SFT on 512 held-out examples.
- Recovered intermediate adapters from ZeRO checkpoints.
- Swept intermediate checkpoints.

Why:

- We needed to know whether the whole new training run was bad or only the final checkpoint was bad.

Conclusion:

- The best intermediate checkpoint, `ckpt-1250`, was still worse than previous SFT on task macro.
- The new SFT was rejected.

### Failure B: Adapter Checkpoint Files Were Empty

What failed:

- Some new-round checkpoint adapter files were empty, so they could not be loaded directly.

How it failed:

- Normal PEFT adapter files did not contain usable tensor data.

What we tried:

- Added checkpoint validation.
- Recovered LoRA adapter tensors from DeepSpeed ZeRO checkpoint state.

Why:

- Without real adapter tensors, inference comparisons are meaningless.
- We needed to distinguish a broken save artifact from actual model quality.

Conclusion:

- Recovery worked.
- It showed the model quality itself had regressed, not just the adapter save path.

### Failure C: Benchmark Config Treated Adapter as Full Checkpoint

Failed job:

`32654981`

What failed:

- The benchmark tried to load the PEFT adapter directory as if it were a full merged model checkpoint.

How it failed:

- Model loading failed before useful inference.

What we tried:

- Updated `benchmarks/vlm_baselines/agvlm_checkpoint_models.yaml`.
- Set the base model to `microsoft/Phi-4-reasoning-vision-15B`.
- Set the trained SFT directory as `adapter_path`, not `checkpoint_path`.

Why:

- A LoRA adapter does not contain the full base model. It must be attached to the base model at load time.

Conclusion:

- The config now correctly loads base model plus adapter.

### Failure D: Cluster Runtime Cache Permission Error

Failed jobs:

- `32655297`
- `32655370`
- `32656485`

What failed:

- The benchmark inherited runtime cache paths under `/run/user/9419`, which the compute job could not use correctly.

How it failed:

- Model load failed due to runtime path permissions.

What we tried:

- Patched the Slurm benchmark wrapper to force these under the job workspace:
  - `TMPDIR`
  - `XDG_RUNTIME_DIR`
  - `TRITON_CACHE_DIR`
  - `TORCHINDUCTOR_CACHE_DIR`

Why:

- GPU jobs should use job-local writable cache directories, not a stale login-node runtime directory.

Conclusion:

- The runtime permission failure was fixed.

### Failure E: PEFT Adapter Dtype Cast With 4-bit Model

Failed jobs:

- `32656509`
- `32674850`

What failed:

- PEFT tried to autocast adapter dtype while the base model was loaded in 4-bit bitsandbytes quantization.

How it failed:

- bitsandbytes rejected dtype casting on a quantized model.

What we tried:

- Passed `autocast_adapter_dtype=False` when attaching the adapter to a 4-bit model.

Why:

- Quantized models cannot be freely cast after loading.
- Adapter attachment should not trigger a dtype cast that the quantized base cannot support.

Conclusion:

- This removed one adapter loading failure.

### Failure F: Phi-4 Remote Code Cast the 4-bit Model After Loading

Failed jobs:

- `32675118`
- `32675451`
- `32675619`

What failed:

- The remote Phi-4 reasoning vision code called `model.to(dtype)` after loading.

How it failed:

- bitsandbytes raised:

`You cannot cast a bitsandbytes model in a new dtype.`

What we tried:

- First tried passing dtype into `from_pretrained`.
- That was not enough, because the remote model code still did a post-load `.to(dtype)`.
- Then loaded Phi-4 reasoning vision through `Phi4ForCausalLMV` directly.
- Added a guard for the invalid post-load dtype cast.

Why:

- The failure was inside model loading code downloaded through `trust_remote_code`.
- We needed to patch only the invalid quantized cast while preserving the real model load.

Conclusion:

- Model loading reached generation after this fix.

### Failure G: Generation Failed With Float vs BFloat16 Mismatch

Failed benchmark:

`32675757`

What failed:

- The benchmark loaded the model and wrote 392 prediction records, but every generation attempt failed.

How it failed:

- Every prediction had this runtime error:

`RuntimeError: expected scalar type Float but found BFloat16`

What we tried:

- Added CUDA autocast during generation for bf16/fp16 inference.

Why:

- The project's other inference path already uses autocast.
- Phi-4 vision components and quantized language components need compatible dtype behavior during generation.

Conclusion:

- The next diagnostic completed without runtime errors.

### Failure H: Immediate EOS Caused Prompt Echo to Be Parsed

Diagnostic job:

`32676418`

What failed:

- The model emitted no new answer tokens on a one-sample diagnostic.
- The fallback decoding path decoded the full prompt, which then got parsed as if it were the model answer.

How it failed:

- The parsed answer became the instruction placeholder:

`<most specific crop issue, disease, pest, or label>`

What we tried:

- Added prompt-echo stripping before parsing outputs.
- Added `--min-new-tokens`.
- Set the SFT benchmark wrapper default to `MIN_NEW_TOKENS=2`.

Why:

- Earlier comparison runs used a min-new-token style inference configuration.
- The benchmark should not treat the prompt itself as the answer.

Conclusion:

- Diagnostic job `32676518` generated a real response with no runtime error.

### Failure I: Model Still Refuses Many Classification Prompts

Successful benchmark:

`32676558`

What failed:

- Infrastructure did not fail, but classification quality was poor.

How it failed:

- `classification_macro_f1 = 0.0`.
- Many classification outputs were refusals such as not being able to identify or diagnose pests from images.

What we tried:

- We did not block the current training launch on classification F1 because the configured gate was based on global task macro, VQA, clarify/respond, invalid rate, and zero runtime failures.

Why:

- The immediate goal was to start the next full SFT only if the active checkpoint and benchmark path were functioning.
- The classification refusal problem is a model behavior issue to address in the new training data/objective.

Conclusion:

- The benchmark gate passed, but classification refusal must be a focus of the new training run and follow-up evaluation.

### Failure J: Full Training Job Failed During Processor Save

Failed job:

`32677813`

What failed:

- The benchmark gate successfully submitted full SFT training.
- The training job launched on 16 L4 GPUs.
- The job then failed when the training script tried to save the processor.

How it failed:

The root traceback was:

`AttributeError: 'Phi4VisionRProcessor' object has no attribute 'chat_template'`

The failure happened at:

`processor.save_pretrained(checkpoint_output_dir)`

in:

`src/agri_vlm/training/sft_trainer.py`

Why this matters:

- This is not a benchmark failure. The benchmark and gate already succeeded.
- This is not the same as the previous empty-adapter regression.
- This is a training artifact-save compatibility issue between the project training script and the Phi-4 reasoning vision processor implementation.

What we know from the logs:

- Environment verification passed.
- The job allocated 16 L4 GPUs.
- The full training step failed with Slurm state `FAILED`, exit code `143:0`.
- The relevant failed Slurm step was `32677813.1`, exit code `1:0`.

What we tried:

- Patched the SFT save path in `src/agri_vlm/training/sft_trainer.py`.
- Added `_save_processor(...)`, which sets missing optional processor attributes, including `chat_template` and `audio_tokenizer`, before saving.
- Replaced the direct call to `processor.save_pretrained(checkpoint_output_dir)` with `_save_processor(processor, checkpoint_output_dir)`.
- Added a focused unit test in `tests/test_sft_trainer.py`.

Validation:

- `pytest -q tests/test_sft_trainer.py` passed.
- `python3 -m py_compile src/agri_vlm/training/sft_trainer.py scripts/train/train_sft.py` passed.
- The same full SFT config was resubmitted as job `32679114`.
- Job `32679114` completed successfully with Slurm state `COMPLETED`, exit code `0:0`.

Saved artifact result:

- Final adapter file exists: `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1/adapter_model.safetensors`
- Adapter size: `1782623880` bytes.
- Adapter validation reported `320` tensors and `320` non-empty tensors.
- Adapter dtype: `torch.bfloat16`.
- Processor/tokenizer artifacts were saved, including `processor_config.json`, `tokenizer_config.json`, `tokenizer.json`, `preprocessor_config.json`, and `chat_template.jinja`.

Conclusion:

- The failure was fixed.
- The new SFT candidate was saved successfully.
- The candidate was then benchmarked and rejected by the promotion gate described in Failure K.

### Failure K: Completed SFT Candidate Failed Promotion Benchmark

Benchmark jobs:

- New completed SFT candidate: `32680255`
- Raw Phi-4 reasoning vision base: `32680256`

Both benchmark jobs completed successfully with Slurm exit code `0:0`.

What failed:

- The newly completed SFT candidate did not beat the previous SFT baseline on the required promotion metrics.
- It improved VQA relaxed accuracy but regressed broader behavior.

Promotion gate result:

`REJECT`

| Required Metric | Previous SFT | New Completed SFT | Preferred Delta | Pass |
| --- | ---: | ---: | ---: | --- |
| Task macro average | `0.228774` | `0.207030` | `-0.021744` | no |
| VQA relaxed accuracy | `0.156000` | `0.212000` | `+0.056000` | yes |
| Clarify/respond macro F1 | `0.530323` | `0.409091` | `-0.121232` | no |
| Invalid predictions | `119` | `193` | `-74` | no |

Full comparison metrics:

| Model | Examples | Invalid | Task Macro | Classification F1 | VQA Relaxed | Clarify F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Phi-4 reasoning vision | `392` | `114` | `0.226303` | `0.000000` | `0.188000` | `0.490909` |
| Previous SFT | `392` | `119` | `0.228774` | `0.000000` | `0.156000` | `0.530323` |
| New completed SFT | `392` | `193` | `0.207030` | `0.000000` | `0.212000` | `0.409091` |

How it failed:

- The new completed SFT often emitted only `Answer:` on classification and VQA prompts.
- This raised invalid predictions from `119` to `193`.
- VQA improved, but the gain was not enough to offset the larger format and clarify/respond regressions.
- Classification remained unusable by exact benchmark metrics for all compared Phi-4 variants.

Example pattern:

- Classification examples that raw Phi-4 and previous SFT refused were often reduced by the new SFT to a bare `Answer:` with no label.
- Some VQA examples improved, such as `No` answers that were formatted as `Answer: No`.
- Clarify/respond quality dropped enough to fail the promotion gate.

Conclusion:

- Do not promote the completed balanced-v2 instructional SFT candidate.
- Do not start RL from this candidate.
- Keep the previous SFT checkpoint active until a new SFT recipe fixes classification format compliance and clarify/respond regression.

## 8. Latest Successful Benchmark and Gate

Benchmark job:

`32676558`

Output directory:

`benchmarks/vlm_baselines/results/agvlm_previous_sft_benchmark_watch_autocast_min2/attempt-1`

Gate report:

`reports/sft_regression_audit/benchmark_gate_then_sft_32676558.md`

Summary:

| Metric | Value |
| --- | ---: |
| Examples | `392` |
| Runtime failures | `0` |
| Failure rate | `0.0` |
| Invalid predictions | `119` |
| Invalid prediction rate | `0.30357142857142855` |
| Task macro average | `0.22877419354838713` |
| Classification macro F1 | `0.0` |
| VQA relaxed accuracy | `0.156` |
| Clarify/respond macro F1 | `0.5303225806451614` |

Interpretation:

- The model is now benchmarkable under the fixed inference path.
- The current active checkpoint is good enough to pass the gate.
- The full next SFT was correctly submitted.
- Classification is still the main weak area.

## 9. Inference Examples Observed

These examples show the kinds of behaviors we saw.

### Example 1: Classification Refusal

Task:

Identify the insect or pest shown in an agricultural image.

Observed output:

`I'm sorry, but I can't assist with identifying or diagnosing specific pests or diseases from images. It's important to consult a professional agronomist or pest control expert for accurate identification and advice on managing agricultural issues.`

Interpretation:

- This is safe-sounding but not useful for the benchmark.
- It fails the classification task because the requested output is a specific label.

### Example 2: VQA Valid Short Answer

Observed output:

`Yes.`

Interpretation:

- This is parseable for yes/no VQA.
- VQA had a low invalid rate in the latest successful benchmark.

### Example 3: Prompt Echo Failure Before Fix

Observed parsed answer before the prompt-echo fix:

`<most specific crop issue, disease, pest, or label><|end|><|assistant|>`

Interpretation:

- The model did not actually answer.
- The benchmark fallback decoded the prompt itself.
- We fixed this by stripping prompt echo and requiring at least 2 new tokens.

### Example 4: New SFT Output Collapse

Observed new-round outputs included:

- `Answer:`
- `plant.`
- `plant disease`
- `::`

Interpretation:

- These are malformed or too generic.
- This was one reason the new balanced-v2 SFT was rejected.

## 10. Current State

As of this report:

| Item | Status |
| --- | --- |
| Previous SFT checkpoint | Active baseline checkpoint |
| New balanced-v2 SFT final | Rejected |
| Best new-round intermediate checkpoint | `ckpt-1250`, diagnostic only, not promoted |
| Benchmark watcher | Completed successfully |
| Successful benchmark job | `32676558` |
| Full training job launched by gate | `32677813` |
| First full training status | Failed during `processor.save_pretrained(...)` |
| Save-path fix | Implemented in `src/agri_vlm/training/sft_trainer.py` |
| Replacement full training job | `32679114` |
| Replacement full training status | Completed successfully, exit code `0:0` |
| New completed SFT candidate | `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1` |
| New completed SFT benchmark job | `32680255`, completed successfully |
| Raw Phi-4 base benchmark job | `32680256`, completed successfully |
| New candidate promotion status | Rejected; keep previous SFT active |

Failed training log:

`logs/slurm/agri-vlm-sft-phi4rv-16g-32677813.out`

Failed training error log:

`logs/slurm/agri-vlm-sft-phi4rv-16g-32677813.err`

Successful replacement training log:

`logs/slurm/agri-vlm-sft-phi4rv-16g-32679114.out`

Successful replacement training error log:

`logs/slurm/agri-vlm-sft-phi4rv-16g-32679114.err`

Replacement checkpoint validation:

`/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1/adapter_validation.json`

Completed SFT benchmark comparison:

`reports/sft_regression_audit/completed_sft_benchmark_comparison_20260519.md`

Completed SFT promotion gate:

`reports/sft_regression_audit/completed_sft_promotion_gate_20260519.md`

Benchmark output:

`benchmarks/vlm_baselines/results/agvlm_previous_sft_benchmark_watch_autocast_min2/attempt-1`

## 11. Why We Did Not Start RL Yet

RL/GRPO should not start from a checkpoint that failed SFT promotion.

The rejected new SFT candidates had:

- Worse task macro than previous SFT.
- More invalid outputs.
- Evidence of output collapse.
- Weak or failed classification behavior.

The latest completed candidate improved VQA relaxed accuracy, but it still failed promotion because task macro, clarify/respond F1, and invalid prediction count regressed.

Starting RL from that checkpoint would likely optimize a degraded policy. The safer sequence is:

1. Establish a completed SFT checkpoint.
2. Verify it with benchmark gates.
3. Promote it only if it beats the previous SFT baseline without unacceptable regressions.
4. Only then consider RL/GRPO.

## 12. Key Lessons

1. Training loss is not enough.

Round 2 training loss decreased, but benchmark behavior got worse.

2. Local reward was not promotion-safe.

The new SFT had better local average reward but worse task metrics.

3. Adapter validation is necessary.

Empty adapter files made early inference attempts unreliable until we recovered ZeRO checkpoint tensors.

4. Benchmark infrastructure needs exact model-loading support.

Large remote-code VLMs, PEFT adapters, 4-bit quantization, and cluster cache paths all interacted in ways that caused failures.

5. The current model still has behavior problems.

The latest benchmark passed the gate, but classification refusal remains unresolved.

6. Artifact save compatibility must be tested directly.

The first full training launch completed the trainer path but failed while saving the processor. Future smoke tests should exercise final artifact saving, not only model loading and the first training steps.

## 13. Recommended Next Steps

### Immediate Post-Rejection Follow-Up

This section changed after benchmark jobs `32680255` and `32680256` completed. The training-save issue is fixed, but the newly completed SFT candidate failed promotion.

1. Keep the previous SFT checkpoint active.
2. Do not start RL/GRPO from the rejected completed SFT candidate.
3. Use the side-by-side examples to repair the next SFT data and prompt recipe.
4. Fix the `Answer:`-only output collapse before another full SFT run.
5. Add or upweight classification examples that require a canonical agricultural label plus evidence.
6. Add a small preflight benchmark gate that checks classification format compliance before spending a full training run.

### Benchmark Review Criteria

1. Run the same SFT benchmark gate on every completed candidate checkpoint.
2. Compare against original Phi-4 reasoning vision, previous SFT, rejected balanced-v2 SFT, and external baselines.
3. Inspect examples for classification refusal, generic labels, empty `Answer:` outputs, and clarify/respond over-triggering.
4. Promote only if the new checkpoint beats previous SFT on task macro and does not regress VQA or invalid output rate.

### For Data and Prompting

1. Add more training examples that explicitly allow agriculture pest/disease identification from images within the project scope.
2. Penalize refusal when the task is a benchmark classification task and the prompt requests a label.
3. Keep safety language for uncertainty, but require a useful agricultural answer when the benchmark asks for one.
4. Avoid over-repeating clarify/respond examples.
5. Keep ground-level RGB agriculture as the default scope.

### For Benchmarks

1. Keep exact label metrics.
2. Also report semantic alias accuracy for pest/disease labels.
3. Separate format compliance from factual correctness.
4. Keep zero-runtime-failure as a hard gate.

## 14. Artifact Index

| Area | Path |
| --- | --- |
| Aggregate SFT and benchmark report | `reports/agvlm_sft_benchmark_aggregate_report.md` |
| Latest monitor setup and outcome | `reports/sft_regression_audit/benchmark_gate_monitor_setup.md` |
| Latest successful gate report | `reports/sft_regression_audit/benchmark_gate_then_sft_32676558.md` |
| Latest successful benchmark outputs | `benchmarks/vlm_baselines/results/agvlm_previous_sft_benchmark_watch_autocast_min2/attempt-1` |
| Full SFT training log for failed job `32677813` | `logs/slurm/agri-vlm-sft-phi4rv-16g-32677813.out` |
| Full SFT training error log for failed job `32677813` | `logs/slurm/agri-vlm-sft-phi4rv-16g-32677813.err` |
| Full SFT replacement log for successful job `32679114` | `logs/slurm/agri-vlm-sft-phi4rv-16g-32679114.out` |
| Full SFT replacement error log for successful job `32679114` | `logs/slurm/agri-vlm-sft-phi4rv-16g-32679114.err` |
| New completed SFT candidate checkpoint | `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1` |
| New completed SFT adapter validation | `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1/adapter_validation.json` |
| New completed SFT benchmark outputs | `benchmarks/vlm_baselines/results/agvlm_new_sft_benchmark_20260519` |
| Raw Phi-4 base benchmark outputs | `benchmarks/vlm_baselines/results/agvlm_phi4_base_benchmark_20260519` |
| Completed SFT comparison report | `reports/sft_regression_audit/completed_sft_benchmark_comparison_20260519.md` |
| Completed SFT pairwise examples | `reports/sft_regression_audit/completed_sft_benchmark_pairwise_20260519.md` |
| Completed SFT promotion gate | `reports/sft_regression_audit/completed_sft_promotion_gate_20260519.md` |
| SFT promotion gate script | `scripts/eval/check_sft_promotion_gate.py` |
| Benchmark gate and submit script | `scripts/hpc/benchmark_gate_then_submit_sft.py` |
| Persistent watcher script | `scripts/hpc/watch_benchmark_until_success_then_launch_sft.slurm` |
| Benchmark model adapter code | `benchmarks/vlm_baselines/model_adapters.py` |
| Benchmark runner | `benchmarks/vlm_baselines/run_baselines.py` |
| Rejected new SFT promotion gate | `reports/sft_regression_audit/new_sft_promotion_gate.md` |
| External VLM baseline report | `benchmarks/vlm_baselines/results/baseline_report_20260516/metrics/summary_table.md` |

## 15. Bottom Line

We first discovered that the new balanced-v2 SFT regressed. We then added stricter validation, recovered broken checkpoint artifacts, compared checkpoints, fixed benchmark infrastructure, and built an automatic benchmark gate.

The latest benchmark now runs successfully and passed the configured gate. Because of that, full SFT job `32677813` was launched automatically. That job exposed a separate processor-save bug. We patched the SFT save path and resubmitted the same full training configuration as job `32679114`, which completed successfully.

We then benchmarked the completed candidate. It improved VQA relaxed accuracy but failed promotion because task macro, clarify/respond F1, and invalid prediction count regressed versus the previous SFT. The next decision point is therefore a new SFT recipe, not RL: fix output format collapse and classification behavior first, then rerun the benchmark gate.
