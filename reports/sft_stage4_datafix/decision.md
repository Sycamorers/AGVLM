# Stage4 Data-Fix Decision

Date: 2026-06-02

## Decision

Run a Stage4 SFT data-fix round from the Stage2 B200 adapter, not from Stage3.

Stage3 should not be promoted over Stage2. It improved VQA and the aggregate macro score, but it collapsed classification outputs to two labels. The evidence points to a data/benchmark alignment problem plus an IP102 weighting problem.

## Root Cause

The Stage3 training manifest used stripped IP102 labels and source-specific closed-label prompts, but the benchmark split still used numeric-prefixed IP102 labels such as `23 corn borer` and had no `classification_label_space` metadata. The benchmark prompt therefore exposed a global mixed label set with numeric IP102 labels.

The SFT classification benchmark also had no PlantVillage classification rows, even though Stage3 trained on PlantVillage disease classification. That left the benchmark dominated by IP102 pest rows and PlantDoc rows.

Stage3 training also overweighted IP102: `52,224` IP102 classification rows versus `19,456` PlantVillage and `14,336` PlantDoc rows.

## Processed Data

New Stage4 data artifacts are non-destructive and keep historical Stage2/Stage3 files intact.

| Artifact | Path | Notes |
| --- | --- | --- |
| Stage4 holdout config | `configs/data/eval_build_stage4_datafix.yaml` | Adds PlantVillage to held-out eval coverage. |
| Stage4 valid holdout | `data/manifests/full/local_holdout_eval_stage4_datafix.decode_valid_images.jsonl` | `25,667` valid rows, `0` invalid images. |
| Stage4 no-overlap train | `data/manifests/full/sft_train_phi4_max3_stage4_no_eval_overlap.jsonl` | `287,619` rows before closed-label balancing. |
| Stage4 raw eval | `data/manifests/full/sft_eval_phi4_max3_stage4_raw_stratified768.jsonl` | `768` rows. |
| Stage4 closed-label train | `data/manifests/full/sft_train_phi4_max3_stage4_closed_label_datafix.jsonl` | `128,330` rows. |
| Stage4 closed-label eval | `data/manifests/full/sft_eval_phi4_max3_stage4_closed_label_stratified768.jsonl` | `768` rows with source-specific label spaces. |
| Stage4 benchmark splits | `benchmarks/vlm_baselines/splits_stage4_datafix/` | `152` val, `616` test, no train/eval overlap. |

## Stage4 Mix

Train rows:

| Task | Rows |
| --- | ---: |
| classification | 46,848 |
| vqa | 50,000 |
| consultation | 25,000 |
| clarify_or_respond | 6,482 |

Classification source mix:

| Source | Rows |
| --- | ---: |
| IP102 | 13,056 |
| PlantDoc | 14,336 |
| PlantVillage | 19,456 |

Eval rows:

| Source | Rows |
| --- | ---: |
| IP102 | 273 |
| PlantDoc | 21 |
| PlantVillage | 105 |
| PlantVillage VQA | 337 |
| MIRAGE clarify/respond | 32 |

## Validation

- Image validation passed for Stage4 holdout: `0` invalid rows.
- Closed-label eval repair attached label spaces for IP102 (`102` labels), PlantDoc (`28`), and PlantVillage (`38`).
- Benchmark readiness passed with `0` errors: `reports/benchmark_status_sft_stage4_datafix_20260602.md`.
- SFT dry-run passed with the Stage4 full training config.
- Rendered SFT format audit had `0` validation failures: `reports/sft_stage4_datafix/sft_train_phi4_max3_stage4_closed_label_datafix_format_audit.md`.
- Target-quality audit only flagged expected short answers: `reports/sft_stage4_datafix/sft_train_phi4_max3_stage4_closed_label_datafix_target_quality.md`.

## Next Run

Use:

```bash
PYTHONPATH=src python3 scripts/train/train_sft.py \
  --model-config configs/model/phi4_reasoning_vision_15b_b200.yaml \
  --train-config configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage4_datafix_preflight.yaml
```

Then the full run:

```bash
PYTHONPATH=src python3 scripts/train/train_sft.py \
  --model-config configs/model/phi4_reasoning_vision_15b_b200.yaml \
  --train-config configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage4_datafix.yaml
```

Benchmark the result against the Stage4 split by setting:

```bash
export AGRI_VLM_SFT_EVAL_MANIFEST=data/manifests/full/sft_eval_phi4_max3_stage4_closed_label_stratified768.jsonl
export AGRI_VLM_SFT_TRAIN_MANIFEST=data/manifests/full/sft_train_phi4_max3_stage4_closed_label_datafix.jsonl
export AGRI_VLM_SFT_SPLIT_SUMMARY=data/manifests/full/sft_train_eval_phi4_max3_stage4_datafix_summary.json
```
