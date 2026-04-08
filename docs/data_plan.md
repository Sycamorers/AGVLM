# Data Plan

## V1 source coverage

SFT core sources:
- PlantVillage
- PlantDoc
- IP102
- PlantVillageVQA
- AgBase resources when placed locally
- MIRAGE resources when placed locally
- Agri-LLaVA resources when placed locally

RL core sources:
- PlantVillage label-grounded classification tasks
- IP102 pest classification tasks
- short-answer PlantVillageVQA tasks
- MIRAGE-MMMT clarify-vs-respond tasks
- MIRAGE-MMST structured consultation tasks
- optional AgBase exact or semi-structured tasks

Evaluation:
- AgMMU
- MIRAGE-MMST
- MIRAGE-MMMT
- local holdout from PlantDoc, IP102, and PlantVillageVQA

## Default source policy

The repo uses manual raw-data slots by default.

Reason:
- dataset licensing and stable source URLs vary
- some widely used mirrors are community-hosted rather than canonical
- the repo should not silently change provenance by picking an arbitrary mirror

Current slot behavior:
- `scripts/data/prepare_manual_dataset_slots.py` creates `README.manual.md` and `MANIFEST.stub.json` for each dataset
- normalization scripts expect the engineer to place approved raw exports inside each slot
- smoke data is synthetic and exists only to exercise the pipeline

## Normalization policy

Every normalized row must include:
- image paths
- chat-style messages
- canonical target fields
- source metadata
- verifier metadata
- reward metadata

Rules:
- preserve original labels when possible
- also store normalized labels
- preserve split provenance
- separate templated supervision from human-authored supervision in metadata
- avoid test-set leakage into SFT or RL manifests

## Local holdout policy

Local holdout is built conservatively:
- only from PlantDoc, IP102, and PlantVillageVQA
- grouped by image identity to avoid near-duplicate leakage across splits
- official benchmark test data remains outside train and RL manifests

## Manual steps

Expected manual steps before full training:
- place approved raw datasets under `data/raw/<dataset_name>/`
- run the matching normalization script
- verify license and usage restrictions for AgBase, MIRAGE, Agri-LLaVA, and AgMMU exports
