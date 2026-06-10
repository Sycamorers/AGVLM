# Stage7 Classification Failure Diagnosis

Date: 2026-06-08

## Diagnosis

Stage7 should not be treated as evidence that classification-only SFT is sufficient. It is worse than Stage5 because free-form label generation collapsed to a source-level prior:

| source | mode prediction | mode rate | accuracy | out-of-label-space |
| --- | --- | ---: | ---: | ---: |
| ip102 | aphids | 100.00% | 4.88% | 0.00% |
| plantvillage | peach bacterial spot | 100.00% | 3.85% | 0.00% |
| rice_disease | to spot | 100.00% | 0.00% | 100.00% |
| plantdoc | peach leaf | 100.00% | 4.76% | 0.00% |
| digigreen_crop_disease | aphids | 100.00% | 0.00% | 100.00% |
| banana_disease | to spot | 100.00% | 0.00% | 100.00% |
| tea_sickness | to spot | 100.00% | 0.00% | 100.00% |

The prompt/target format was not the main mismatch: Stage7 training uses `classification_label_only` prompts and bare label targets. The benchmark also used label-only classification prompting. The failure is therefore not just an `Answer:` wrapper issue.

Training loss improved, but generated validation accuracy stayed at zero at every saved validation checkpoint. This is a teacher-forced token-loss improvement without usable closed-label generation.

## Current Risks

- Free-form generation is the wrong primary diagnostic for closed-label classification because it allows empty strings, fragments, and labels outside the candidate set.
- Stage7 validation is too narrow: the validation manifest only covers IP102, rice, and tea, and training-time generation metrics use only the first 96 examples.
- One mixed adapter across very different label spaces is likely causing task/source interference.
- The model may be learning label token priors rather than image-discriminative decision boundaries.

## What To Do Next

Do not launch another SFT run yet. Run an evaluation-only constrained classification benchmark first.

This benchmark keeps the current Stage7 model and image inputs, but restricts classification generation to the source-specific allowed label set. It answers the key question:

- If constrained accuracy improves a lot, the main failure is free-form decoding/output selection.
- If constrained accuracy is still near chance, the main failure is visual discrimination/task learning/data strategy.

The code now supports:

```bash
python benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split test \
  --split-dir benchmarks/vlm_baselines/splits_stage5_datafix \
  --batch-size 1 \
  --max-new-tokens 0 \
  --min-new-tokens 0 \
  --dtype bf16 \
  --quantization 4bit \
  --device cuda:0 \
  --output-dir benchmarks/vlm_baselines/results/<run_tag> \
  --model-key agvlm_phi4_sft_stage7_label_only_classification_b200_candidate \
  --classification-decode-mode constrained
```

Use the Slurm wrapper:

```bash
sbatch scripts/hpc/run_stage7_constrained_classification_benchmark.slurm
```

I did not submit it yet because job `34118542` is already running under this user with one GPU and 512G memory. To avoid resource interference, wait until that job exits before submitting this one.

## Decision Tree After The Constrained Benchmark

1. Constrained benchmark much better than Stage7 free generation:
   - Keep label-only data format.
   - Add constrained decoding or candidate scoring as the classification inference path.
   - Do not spend more SFT time just to fix wrappers.

2. Constrained benchmark still bad:
   - Stop mixed-source classification LoRA.
   - Run a micro-overfit sanity test on one source and 5-10 classes.
   - Then train source/task-specific adapters with balanced validation for every source.

3. Micro-overfit cannot reach high train accuracy:
   - Audit image loading, collator masking, assistant target placement, adapter loading, and Phi-4 vision gradient path before any larger training.

4. Micro-overfit succeeds but held-out accuracy is poor:
   - The issue is data/label-space size/generalization.
   - Add more examples per class and train per-source adapters, not one heterogeneous adapter.
