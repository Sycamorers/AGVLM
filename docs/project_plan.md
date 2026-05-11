# Project Plan

## V1 Scope

Build and evaluate an agriculture-focused VLM for ground-level RGB consultation. V1 covers crop disease, pest, symptom understanding, short VQA, management guidance, and clarify-vs-respond behavior. It should not default to broad generic VLM behavior.

## Current Stage

SFT is the active training stage. The current active path is Phi-4 reasoning vision on the full max-3-image agricultural SFT split. That server run should not be disrupted.

This benchmark update prepares code, configs, scripts, docs, and dry-run checks only. Full benchmark runs should wait until the intended checkpoints are ready.

## Stage 0: Data and Manifests

Inputs include PlantVillage, PlantDoc, IP102, PlantVillageVQA, AgBase/AgMMU-style consultation data, MIRAGE-style clarify/consultation data, and any documented manual sources. Dataset steps that are gated, licensed, or manual must remain explicit in docs and configs.

Split policy:

- avoid sample-id and image-group train/eval overlap
- keep public test data out of training if it is used as benchmark data
- preserve phase labels so SFT and RL results are not mixed

## Stage 1: SFT

Purpose:

- teach agricultural task formats
- improve visual-language grounding for crop/pest/disease labels
- learn short-answer and consultation style
- establish the checkpoint that RL must initialize from

Inputs:

- `data/manifests/full/sft_train_phi4_max3_no_eval_overlap.jsonl`

Output:

- completed SFT checkpoint or LoRA adapter

Benchmark:

- external baselines and completed SFT model on `sft_benchmark`
- split files under `benchmarks/vlm_baselines/splits/sft_*_manifest.jsonl`

Success criteria:

- better agriculture classification and short VQA metrics than generic baselines
- low invalid output rate
- stable multi-image behavior
- no train/eval leakage

## Stage 2: RL / GRPO

Purpose:

- improve reward-verifiable behavior after SFT
- enforce answer formatting and structured consultation sections
- improve clarify-vs-respond decisions
- increase management keyword coverage where expected
- improve uncertainty and overconfidence control
- avoid forbidden claims

Input:

- completed SFT checkpoint or adapter
- reward-verifiable RL train manifest

Output:

- completed RL checkpoint or adapter

Benchmark:

- external baselines, completed SFT model, and completed RL model on `rl_benchmark`
- split files under `benchmarks/vlm_baselines/splits/rl_*_manifest.jsonl`

Success criteria:

- RL improves structured/reward-verifiable diagnostics over SFT
- RL does not significantly degrade core classification/VQA metrics
- RL lowers unsafe or overconfident claim rates
- RL improves clarify decision metrics

## Stage 3: Results

Export summary tables from `benchmarks/vlm_baselines/results/metrics/summary_table.csv`, training curves from run directories, and final reports under `reports/` and `outputs/artifacts/`. Keep SFT and RL benchmark rows phase-tagged in all final tables.
