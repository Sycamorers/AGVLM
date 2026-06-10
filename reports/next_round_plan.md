# Recommended Next-round Plan

## Direct assessment

The current setup is flawed for diagnosing classification. One mixed LoRA adapter may be acceptable for broad agriculture assistant behavior, but it is not the right default experiment for fixing many source-specific, high-cardinality classification tasks. The first fixes should be evaluation and format standardization, not more blind SFT.

## Before retraining

- Freeze the benchmark split and rerun metrics with the robust parser.
- Choose one classification output contract: preferably label-only for classification adapters, or `Answer: <label>` with parser-based scoring.
- Separate classification metrics from VQA, consultation, and clarify/respond metrics.
- Add per-source confusion matrices and source prediction-mode collapse checks to every benchmark.
- Balance low-resource classes and document any synthetic or licensed/manual additions.

## Data target

For a real classification repair run, target at least 50-100 clean examples per class for small label spaces, and 100-300 per class for visually subtle or high-cardinality sources such as IP102. Classes below 20 examples should be treated as diagnostic-only unless augmented or merged into a scoped label space.

## Experiments

A. Evaluation-only fix: keep the current model, apply robust parsing, recompute strict and normalized metrics, and measure metric/output mismatch.

B. Format-standardized SFT: keep data content, rewrite classification targets to canonical labels only, retrain LoRA, compare exact and normalized metrics.

C. Task-specific classification LoRA: train only classification-style data with strict label-only output and compare against the mixed Stage5 adapter.

D. Data scaling test: add or synthesize balanced examples for underrepresented classes, use balanced sampling, and measure low-resource class recall.

E. General vs specialized adapter comparison: compare one mixed-domain LoRA against source/task-specific adapters with identical benchmark splits.

## Training configs to try

- Classification-only adapter: smaller LR sweep around `2e-7`, `5e-7`, `1e-6`; keep deterministic decoding and label-only targets.
- Lower LoRA capacity ablation: compare r=64/128/256 to test whether high-rank mixed SFT is memorizing style/collapse patterns.
- Add eval generation metrics every checkpoint and promote only on normalized accuracy, macro F1, and collapse checks.
- Keep consultation/VQA SFT separate or stage it before classification, then optionally use DPO/GRPO only after SFT format and metrics are stable.

## Prompt/constrained decoding

Prompt engineering and constrained decoding can help classification immediately. For closed label spaces, constrained decoding over labels or option letters should be tested before more training because it directly addresses invalid/out-of-format outputs without changing weights.
