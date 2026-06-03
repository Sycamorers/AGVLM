# Stage5 Dataset Research Update

Date: 2026-06-03

## Quota Guardrail

`blue_quota` before and after the new downloads reported group `hmedeiros` at `12.28T` used on `blue2`. The requested guardrail is `14T`, leaving about `1.72T` of headroom after this update.

## Added Datasets

| Dataset | HF Repo | License | Rows | Classes | Reason |
| --- | --- | --- | ---: | ---: | --- |
| `banana_disease` | `as-cle-bert/banana-disease-classification` | `cc-by-4.0` | 777 | 7 | Adds banana disease and pest coverage not represented in PlantVillage/PlantDoc. |
| `tea_sickness` | `yunusserhat/tea_sickness_dataset` | `cc-by-4.0` | 885 | 8 | Adds tea leaf disease coverage with explicit train/validation/test splits. |

Both were downloaded in `full` mode, normalized to `data/interim/full/`, and decode-validated with `0` invalid rows.

## New Artifacts

| Artifact | Rows |
| --- | ---: |
| `data/raw/banana_disease/full/records.jsonl` | 777 |
| `data/raw/tea_sickness/full/records.jsonl` | 885 |
| `data/interim/full/banana_disease.jsonl` | 777 |
| `data/interim/full/tea_sickness.jsonl` | 885 |
| `data/manifests/full/banana_disease.decode_valid_images.jsonl` | 777 |
| `data/manifests/full/tea_sickness.decode_valid_images.jsonl` | 885 |
| `data/manifests/full/banana_disease.decode_invalid_images.jsonl` | 0 |
| `data/manifests/full/tea_sickness.decode_invalid_images.jsonl` | 0 |

## Rejected Or Deferred Candidates

| Candidate | Decision | Reason |
| --- | --- | --- |
| `SyedNazmusSakib/PlantInquiryVQA` | Deferred | Good VQA fit, but CSV references `24,964` unique image IDs while the repo currently exposes `10,000` image files; `14,964` image IDs were missing in the probe. Do not include until the full image corpus is available or a documented partial-ingestion policy is added. |
| `Saon110/bd-crop-vegetable-plant-disease-dataset` | Rejected for now | HF reports it as gated; no automatic public download path is available. |
| `Engineer101/cassava-leaf-disease-classification` | Rejected for now | HF builder exposed `image` only and no `label` feature, so the current generic classification materializer cannot build targets. |
| `aquib1011/maize-leaf-disease` | Deferred | Relevant RGB crop disease dataset, but no explicit license was available from the HF metadata probe. |
| `anthony2261/paddy-disease-classification` | Deferred | Relevant RGB rice disease dataset, but no explicit license was available from the HF metadata probe. |
| Broad plant species or aerial/field-boundary datasets | Rejected | Outside V1 scope, which is ground-level RGB crop disease, pest, symptom, and consultation behavior. |

## Config Changes

- Added `default_crop` to dataset registry entries so single-crop HF classification datasets can preserve crop metadata without hardcoded dataset-name checks.
- Added `banana_disease` and `tea_sickness` to Stage5 SFT build and holdout eval configs.
- Added modest Stage5 closed-label caps: `64` rows per label for each new source.
