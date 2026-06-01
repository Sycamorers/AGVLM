# Benchmark Gate Monitor Setup

Date: 2026-05-19

## Active Jobs

| Purpose | Slurm Job | Status |
| --- | ---: | --- |
| Persistent benchmark watchdog | `32676538` | Completed successfully |
| Successful benchmark attempt | `32676558` | Completed successfully on `hpg-turin` |
| Full SFT training | `32677813` | Submitted by gate, then failed during processor save |
| Replacement full SFT training | `32679114` | Completed successfully after save-path patch |
| New completed SFT benchmark | `32680255` | Completed successfully |
| Raw Phi-4 base benchmark | `32680256` | Completed successfully |

## Final Gate Result

The corrected benchmark attempt completed with no runtime failures and passed all configured gates, so the watchdog submitted full SFT job `32677813`.

The gate/submission mechanism succeeded. The submitted full SFT job later failed in the training save path with:

`AttributeError: 'Phi4VisionRProcessor' object has no attribute 'chat_template'`

The save path was patched in `src/agri_vlm/training/sft_trainer.py` so SFT processor saving adds missing optional processor attributes before calling `save_pretrained(...)`. The same full training configuration was resubmitted as Slurm job `32679114`, which completed successfully with exit code `0:0`.

The completed SFT candidate was then benchmarked and rejected by the promotion gate. It improved VQA relaxed accuracy, but regressed task macro, clarify/respond F1, and invalid prediction count versus the previous SFT baseline.

| Gate | Actual | Requirement | Pass |
| --- | ---: | ---: | --- |
| `num_examples` | `392` | `>= 392` | yes |
| `failure_rate` | `0.0` | `<= 0.0` | yes |
| `invalid_prediction_rate` | `0.30357142857142855` | `<= 0.50` | yes |
| `task_macro_average` | `0.22877419354838713` | `>= 0.13` | yes |
| `vqa.relaxed_accuracy` | `0.156` | `>= 0.12` | yes |
| `clarify_or_respond.macro_f1` | `0.5303225806451614` | `>= 0.20` | yes |

Output paths:

- Benchmark outputs: `benchmarks/vlm_baselines/results/agvlm_previous_sft_benchmark_watch_autocast_min2/attempt-1`
- Gate report: `reports/sft_regression_audit/benchmark_gate_then_sft_32676558.md`
- Gate JSON: `reports/sft_regression_audit/benchmark_gate_then_sft_32676558.json`
- Successful replacement training log: `logs/slurm/agri-vlm-sft-phi4rv-16g-32679114.out`
- New completed SFT checkpoint: `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1`
- Adapter validation: `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-balanced-v2-instructional-full-turin16-batch1/adapter_validation.json`
- New completed SFT benchmark outputs: `benchmarks/vlm_baselines/results/agvlm_new_sft_benchmark_20260519`
- Raw Phi-4 base benchmark outputs: `benchmarks/vlm_baselines/results/agvlm_phi4_base_benchmark_20260519`
- Completed SFT comparison report: `reports/sft_regression_audit/completed_sft_benchmark_comparison_20260519.md`
- Completed SFT promotion gate: `reports/sft_regression_audit/completed_sft_promotion_gate_20260519.md`

## Completed SFT Benchmark Result

Decision: **do not promote** `agvlm_phi4_sft_balanced_v2_instructional_completed`.

| Model | Examples | Invalid | Task Macro | Classification F1 | VQA Relaxed | Clarify F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Phi-4 reasoning vision | `392` | `114` | `0.226303` | `0.000000` | `0.188000` | `0.490909` |
| Previous SFT | `392` | `119` | `0.228774` | `0.000000` | `0.156000` | `0.530323` |
| New completed SFT | `392` | `193` | `0.207030` | `0.000000` | `0.212000` | `0.409091` |

Promotion gate failures:

- `task_macro_average`: `0.207030` candidate versus `0.228774` previous SFT.
- `clarify_or_respond.macro_f1`: `0.409091` candidate versus `0.530323` previous SFT.
- `num_invalid_predictions`: `193` candidate versus `119` previous SFT.

Only the VQA relaxed accuracy requirement passed: `0.212000` candidate versus `0.156000` previous SFT.

## Gate Behavior

The monitor job runs `scripts/hpc/monitor_benchmark_then_launch_sft.slurm`, which calls:

`scripts/hpc/benchmark_gate_then_submit_sft.py`

The active watchdog runs:

`scripts/hpc/watch_benchmark_until_success_then_launch_sft.slurm`

It submits benchmark attempts under:

`benchmarks/vlm_baselines/results/agvlm_previous_sft_benchmark_watch_autocast_min2/attempt-*`

When an attempt completes successfully, it reads metrics from that attempt's `metrics/` directory and runs the same gate.

Full SFT is submitted only if all configured gates pass:

| Gate | Requirement |
| --- | ---: |
| `num_examples` | `>= 392` |
| `failure_rate` | `<= 0.0` |
| `invalid_prediction_rate` | `<= 0.50` |
| `task_macro_average` | `>= 0.13` |
| `vqa.relaxed_accuracy` | `>= 0.12` |
| `clarify_or_respond.macro_f1` | `>= 0.20` |

If the benchmark fails, the watchdog retries. If a benchmark completes but records runtime failures above the gate, the watchdog retries. If the benchmark completes cleanly but misses quality gates, no training job is submitted.

## Training Submission

If the gate passes, the monitor submits:

`scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm`

with:

| Config | Path |
| --- | --- |
| Train config | `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_balanced_v2_instructional_full.yaml` |
| Preflight config | `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_balanced_v2_instructional_preflight.yaml` |
| Model config | `configs/model/phi4_reasoning_vision_15b_turin_24g.yaml` |
| Data config | `configs/data/sft_train_eval_phi4_max3.yaml` |

The monitor writes its decision report to:

- `reports/sft_regression_audit/benchmark_gate_then_sft_<successful_benchmark_job_id>.json`
- `reports/sft_regression_audit/benchmark_gate_then_sft_<successful_benchmark_job_id>.md`

## Notes

The benchmark Slurm wrapper was updated to force `TMPDIR`, `XDG_RUNTIME_DIR`, `TRITON_CACHE_DIR`, and `TORCHINDUCTOR_CACHE_DIR` under the job workspace; this avoids the prior `/run/user/9419` permission failure during model load.

Earlier replaced attempts:

- `32656485` failed because `TMPDIR` still inherited `/run/user/9419`.
- `32656486` was canceled before it could act on that failed benchmark.
- `32656509` failed because PEFT adapter attachment tried to autocast adapter dtype on a 4-bit bitsandbytes model.
- `32656510` correctly rejected training because `32656509` failed.
- `32674850` failed with the same bitsandbytes dtype-cast error.
- `32674851` correctly rejected training because `32674850` failed.
- `32675118` failed after adding `torch_dtype`; Transformers still took a post-load cast path.
- `32675119` correctly rejected training because `32675118` failed.
- `32675451` diagnostic failed and exposed the remote Phi-4 reasoning vision post-load `model.to(dtype)` cast.
- `32675619` failed because it launched before the direct Phi class/dtype guard patch was in place.
- `32675757` completed but every prediction failed with `expected scalar type Float but found BFloat16`; the gate correctly rejected training.
- `32676418` diagnostic completed without runtime errors after adding generation autocast, but immediate EOS caused prompt echo to be parsed as an invalid answer.
- `32676518` diagnostic completed with `min_new_tokens=2`, no runtime errors, and a real generated response.

Current fix:

- `benchmarks/vlm_baselines/model_adapters.py` now passes `autocast_adapter_dtype=False` when attaching PEFT adapters to 4-bit models.
- For 4-bit base load, `benchmarks/vlm_baselines/model_adapters.py` now passes `dtype` directly instead of deprecated `torch_dtype`.
- `benchmarks/vlm_baselines/model_adapters.py` now loads Phi-4 reasoning vision through `Phi4ForCausalLMV` directly and guards the invalid bitsandbytes post-load dtype cast in the remote model code.
- `benchmarks/vlm_baselines/model_adapters.py` wraps generation in CUDA autocast for bf16/fp16 inference.
- `benchmarks/vlm_baselines/model_adapters.py` strips prompt echo before parsing outputs.
- `benchmarks/vlm_baselines/run_baselines.py` and `benchmarks/vlm_baselines/slurm/run_sft_benchmark_24gb.sbatch` now support `min_new_tokens`; the SFT benchmark wrapper defaults to `MIN_NEW_TOKENS=2`.
- `scripts/hpc/watch_benchmark_until_success_then_launch_sft.slurm` retries completed benchmark attempts when runtime failures exceed the configured failure-rate gate.

Training-save follow-up:

- Failed job `32677813` reached final SFT save and failed at direct `processor.save_pretrained(...)`.
- `tests/test_sft_trainer.py` now includes a processor save regression test for missing optional Transformers processor attributes.
- `pytest -q tests/test_sft_trainer.py` passed.
- `python3 -m py_compile src/agri_vlm/training/sft_trainer.py scripts/train/train_sft.py` passed.
- Replacement job `32679114` resumed from `checkpoint-2813`, reached final save, and completed successfully.
- The saved adapter has `320` tensors, `320` non-empty tensors, and `torch.bfloat16` weights.
- The saved checkpoint includes processor/tokenizer artifacts, including `chat_template.jinja`.
