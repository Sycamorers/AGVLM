# Data Plan

## Source Policy

The repo now uses a hybrid source model:

- automatic partial download when a selective public Hugging Face path is available
- manual subset-tagged raw-data slots when selective remote download is not practical

Default acquisition mode:
- `download_mode: partial`
- `sample_fraction: 0.1`
- subset tag: `partial_10pct`

Full rerun mode:
- `download_mode: full`
- `sample_fraction: 1.0`
- subset tag: `full`

## Dataset Coverage

Automatic partial download:
- PlantVillage
- PlantDoc
- PlantVillageVQA
- MIRAGE

Manual staging:
- IP102
- AgBase resources
- Agri-LLaVA / Agri-400K

## Storage Layout

- raw: `data/raw/<dataset_name>/<subset_tag>/`
- normalized: `data/interim/<subset_tag>/<dataset_name>.jsonl`
- merged manifests: `data/manifests/<subset_tag>/`

Every normalized row preserves:
- source dataset
- split
- original labels
- normalized labels when available
- image paths
- license metadata when known
- subset tag and download provenance

## Holdout Policy

The local holdout stays conservative:

- source datasets: PlantDoc, IP102, PlantVillageVQA
- grouped by image identity when available
- official benchmark test data stays out of SFT and RL manifests

## Current Constraints

- IP102, AgBase resources, and Agri-LLaVA still need manual staging
- PlantDoc currently reduces multi-label annotations to one deterministic primary label per image
