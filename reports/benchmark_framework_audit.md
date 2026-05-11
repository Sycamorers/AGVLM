# Benchmark Framework Audit

Date: 2026-05-11

## Scope

This audit covers the inference-only benchmark framework under `benchmarks/vlm_baselines/`, the benchmark status/export scripts, and the project docs relevant to a two-stage SFT/RL evaluation plan. It does not cover SFT training code behavior.

## Current Splits

- Existing legacy split files: `benchmarks/vlm_baselines/splits/val_manifest.jsonl` and `benchmarks/vlm_baselines/splits/test_manifest.jsonl`.
- New SFT benchmark split files:
  - `benchmarks/vlm_baselines/splits/sft_val_manifest.jsonl`: 120 rows.
  - `benchmarks/vlm_baselines/splits/sft_test_manifest.jsonl`: 392 rows.
  - Source: `data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl`.
  - The full held-out SFT eval manifest has 512 rows; the new val/test files are disjoint aliases of that held-out set.
- New RL benchmark split files:
  - `benchmarks/vlm_baselines/splits/rl_val_manifest.jsonl`: 369 rows.
  - `benchmarks/vlm_baselines/splits/rl_test_manifest.jsonl`: 1,573 rows.
  - Source: `data/manifests/full/rl_local_holdout_eval.jsonl`, with rows that shared exact sample IDs or image groups with `data/manifests/full/rl_manifest.jsonl` filtered out.
- New split reports:
  - `benchmarks/vlm_baselines/splits/benchmark_split_report.json`
  - `benchmarks/vlm_baselines/splits/benchmark_split_report.md`

## Supported Models

External baseline config currently supports:

- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- `google/paligemma2-3b-mix-448`
- `microsoft/Phi-4-multimodal-instruct`
- `allenai/Molmo2-4B`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- `Qwen/Qwen2.5-VL-3B-Instruct`

New checkpoint config support:

- `agvlm_phi4_sft_completed`
- `agvlm_phi4_rl_completed`

The checkpoint entries are placeholders by default. They are warnings in readiness reports and fatal when selected for a benchmark run until real paths are provided.

## Implemented Metrics

Classification / label diagnosis:

- top-1 accuracy
- macro-F1
- weighted-F1
- balanced accuracy
- per-class precision, recall, F1, and support
- confusion matrix
- invalid-output rate
- missing-answer rate
- out-of-label-space rate
- parse status counts

Short VQA:

- exact match
- normalized exact match
- relaxed accuracy
- token-F1
- yes/no accuracy for yes/no subsets
- numeric relaxed accuracy for numeric subsets
- answer containment diagnostic only
- invalid and missing answer rates

Clarify-or-respond:

- decision accuracy
- clarify precision, recall, F1
- respond precision, recall, F1
- macro-F1
- confusion matrix
- over-clarification and under-clarification rates
- invalid decision rate
- empty clarifying question / empty answer rates

Consultation / open-ended management:

- structured section compliance
- required section compliance overall and by section
- management keyword coverage
- forbidden claim rate
- unsafe or overconfident claim rate
- uncertainty compliance
- follow-up question presence
- answer length statistics
- repetition rate
- token-F1 only as a diagnostic

Aggregate:

- overall example count
- failure rate
- invalid prediction rate
- task macro average across task families
- per-task metrics
- per-source-dataset metrics
- per-phase summaries
- optional bootstrap confidence intervals through `--bootstrap-samples`

## Task Types Handled

- `classification`
- `vqa`
- `clarify_or_respond`
- `consultation`

The parser recognizes `Answer:`, `Decision:`, and line-start consultation headers: `Diagnosis:`, `Evidence:`, `Uncertainty:`, `Management:`, and `Follow-up:`.

## Appropriateness Assessment

- Classification metrics are appropriate for closed label diagnosis because they report class imbalance-sensitive macro and balanced scores, not only accuracy.
- Short VQA metrics are appropriate for short deterministic answers, yes/no questions, and numeric answers. Long contradictory answers are not rewarded as full yes/no relaxed matches.
- Clarify-or-respond metrics are appropriate because both decision classes are reported separately.
- Consultation metrics are deterministic proxies only. They evaluate structure, keyword coverage, uncertainty, forbidden claims, and repetition, but cannot fully verify agronomic correctness.

## Checkpoint Benchmark Readiness

The previous framework could run external Hugging Face model names but had no strict project checkpoint config or phase-aware checkpoint metadata. It now supports:

- external baselines
- raw/base HF entries if configured as `checkpoint_type: base`
- completed SFT checkpoints or LoRA adapters
- completed RL checkpoints or LoRA adapters

RL entries must record the completed SFT checkpoint used to initialize RL.

## Leakage Risk Assessment

- SFT: exact sample-id overlap and image-group overlap with `sft_train_phi4_max3_no_eval_overlap.jsonl` are currently zero in the new split report.
- RL: the raw local holdout had image-group overlap with the RL train manifest. The builder now filters overlapping rows before producing RL benchmark val/test manifests. Current exact and group overlap are zero after filtering.
- Prompt leakage: SFT prompt leakage count is zero. RL has clarify/decision-word heuristic prompt hits because prompts contain words like clarify/respond in formatting instructions; ground-truth leakage count is zero.
- Public test data: current reports do not flag public test rows in either phase.

## Prompt and Multi-Image Handling

- Benchmark prompts now use a common task-family format with explicit `Answer:` or `Decision:` fields.
- Consultation prompts use required line-start section headers.
- Multi-image distribution is reported. SFT contains 489 one-image, 9 two-image, and 14 three-image rows. RL benchmark is single-image after the RL V1 scope and leakage filter.
- Single-image model adapters use first-image policy and record that limitation.

## Summary Table Phase Awareness

The previous summary table was split-only. The new summary table rows include:

- phase
- split
- model name/key
- checkpoint type
- base model
- adapter/checkpoint path
- dtype and quantization
- generation config
- manifest and prediction paths
- metrics path

Rows for `sft_benchmark` and `rl_benchmark` remain separate.

## Missing Before This Update

For a fair SFT benchmark:

- explicit `sft_benchmark` phase labels
- strict failure when the SFT held-out manifest is absent
- project checkpoint config and validation
- `Answer:` parsing for labels and VQA
- balanced accuracy and parse-status diagnostics
- phase-aware summary tables

For a fair RL benchmark:

- explicit `rl_benchmark` phase labels
- RL local-holdout benchmark split separated from RL train
- leakage filter against RL train image groups
- SFT-vs-RL checkpoint comparison metadata
- consultation and reward-aligned deterministic diagnostics
- checkpoint validation preventing raw-base RL evaluation as an RL checkpoint

## Remaining Limitations

- The RL benchmark local holdout required filtering 2,154 rows due to image-group overlap with the RL train manifest, leaving 1,942 rows. This is fairer but should be reported in final results.
- Consultation metrics are deterministic proxies and should not be presented as full agronomic correctness.
- No LLM-as-judge protocol is implemented.
- No full benchmark was run during this preparation.
