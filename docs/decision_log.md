# Decision Log

## 2026-04-08 Dataset Audit

Already working before this pass:
- unified JSONL schema, manifest builders, and smoke/unit-test scaffolding
- subset-agnostic training and evaluation entrypoints
- B200-oriented multi-GPU training path

Missing before this pass:
- dataset registry was manual-slot only
- no partial-vs-full download policy
- no subset-tagged raw/interim/manifest layout
- no HiPerGator-oriented data-prep scripts
- no dataset summary report

## 2026-04-08 This Upgrade Pass

Changed:
- added a hybrid dataset registry with public and manual source modes
- standardized dataset outputs around `partial_10pct` and `full` subset tags
- added Hugging Face materializers for PlantVillage, PlantDoc, PlantVillageVQA, and MIRAGE
- kept manual-slot fallbacks for IP102, AgBase resources, and Agri-LLaVA where selective 10 percent remote download is not practical
- added `scripts/data/normalize_all.py` and subset-tag-aware manifest builders and reports
- added HiPerGator helpers under `scripts/hpc/`
- rewrote `README.md` and `TODO.md` around the new data workflow

## 2026-04-08 Deferred

Deferred on purpose:
- PlantDoc still uses a deterministic single-label reduction for multi-label annotations
- manual staging is still required for IP102, AgBase resources, and Agri-LLaVA

## 2026-05-06 Training Cleanup And Next Path

Changed:
- cancelled B200 job `31951103` after confirming it was stalled in inline distributed generation evaluation after step-500 loss eval
- excluded the AGBASE-disjoint continuation from the next-stage base because validation quality degraded and no `checkpoint-500` was written
- cleaned local `logs/` and `outputs/` plus obsolete Orange SFT checkpoints and failed run directories
- retained only the balanced Llama 4 Scout adapter needed for the next stage:
  - `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft`
- added B200 full max3 probe/full configs and a Slurm wrapper that rebuilds no-overlap train/eval manifests at launch

Decision:
- continue from the retained balanced adapter rather than retraining from scratch or resuming the AGBASE-disjoint run
- keep inline generation metrics disabled for large SFT jobs and run generation evaluation separately on selected checkpoints

## External Verification Used

- PlantVillage on Hugging Face:
  - https://huggingface.co/datasets/GVJahnavi/PlantVillage_dataset
- PlantVillageVQA on Hugging Face:
  - https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA
- MIRAGE on Hugging Face:
  - https://huggingface.co/datasets/MIRAGE-Benchmark/MIRAGE
- Agri-LLaVA VQA Bench on Hugging Face:
  - https://huggingface.co/datasets/Agri-LLaVA-Anonymous/Agri_LLaVA_VQA_Bench
