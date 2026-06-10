# Micro Banana Overfit Diagnosis

Date: 2026-06-08T12:20:10-04:00

## Run

- Slurm job: `34124840`
- Train config: `configs/train/sft_phi4_reasoning_vision_15b_b200_1gpu_micro_banana_overfit_label_only.yaml`
- Data config: `configs/data/sft_micro_banana_overfit_label_only_phi4_max3.yaml`
- Adapter: `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-micro-banana-overfit-label-only-b200-1gpu`
- Train benchmark: `benchmarks/vlm_baselines/results/micro_banana_overfit_34124840/train_constrained`
- Heldout benchmark: `benchmarks/vlm_baselines/results/micro_banana_overfit_34124840/eval_constrained`

## Data

- Train rows: `48`
- Heldout rows: `15`
- Selected labels: `['banana healthy leaf', 'black sigatoka', 'bract mosaic virus', 'insect pest', 'moko disease', 'yellow sigatoka']`

## Metrics

| split | examples | top1 | macro F1 | weighted F1 | balanced acc | OOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 48 | 16.67% | 4.76% | 4.76% | 16.67% | 0.00% |
| heldout | 15 | 13.33% | 3.92% | 3.14% | 16.67% | 0.00% |

## Interpretation

The fresh LoRA did not memorize the tiny banana train split.
Before another broad SFT run, debug image loading, assistant-label masking, adapter attachment, and Phi-4 vision gradient flow.
