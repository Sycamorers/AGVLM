# RLFT Design

## Scope

RLFT in this repository means rule-based GRPO post-training for reward-verifiable agricultural VLM behavior. V1 remains scoped to ground-level RGB agricultural consultation, classification, VQA, management, uncertainty, and clarify-vs-respond behavior.

This is not full RLHF. The current implementation does not train or load a learned reward model, and it does not implement PPO. Rewards are deterministic or semi-deterministic functions over the model completion plus manifest verifier fields. A preference-data schema and pairwise export scaffold are present for later learned reward work, but they are optional and inactive by default.

## Why GRPO

The target behaviors have verifiable pieces: labels, accepted answers, required structured sections, clarification decisions, uncertainty flags, management keywords, and forbidden claims. TRL `GRPOTrainer` supports custom `reward_funcs`, so the repo can compute one scalar reward per completion without preference pairs or a learned reward model.

GRPO is used only after SFT. Non-dry-run GRPO requires `sft_checkpoint_path` to point to an existing completed SFT checkpoint or adapter. The trainer rejects placeholders, missing paths, and the raw base model name `microsoft/Phi-4-reasoning-vision-15B`.

## Reward Modules

- `exact_match`: rewards exact normalized matches against target answers or accepted answers.
- `normalized_label`: rewards canonical agricultural label matches after label normalization.
- `synonym_match`: rewards acceptable synonym groups when exact labels differ.
- `structured_format`: rewards required consultation sections such as Diagnosis, Evidence, Uncertainty, Management, and Follow-up only when each section has meaningful non-empty content. Heading-only or repeated empty sections are penalized.
- `uncertainty_calibration`: rewards uncertainty language when the verifier marks evidence as insufficient and the uncertainty statement is grounded in image ambiguity, missing evidence, or a need for more information.
- `clarify_vs_respond`: rewards the correct high-level decision. JSON `{"decision": "clarify"}` or `{"decision": "respond"}` is honored when present; plain clarification questions are detected only when they are not substantive answers.
- `management_coverage`: rewards unique expected management keywords or steps only when they appear in meaningful answer context. Repetition, keyword lists, and very long repetitive completions are capped or penalized.
- `hallucination_penalty`: penalizes configured forbidden claims, target-label contradictions, unsupported definitive answers when clarification is expected, overconfidence, unsupported chemical/dosage/safety advice, fabricated visual evidence, unsafe recommendations, and crop/disease mismatches when metadata is available.
- `preference_proxy`: optional scaffold for pairwise preference rows. It is not used by default and is not a learned reward model.

The composite reward function is exposed through `make_trl_reward_function()` and accepts `prompts`, `completions`, `task_type`, `target_json`, `verifier_json`, `reward_meta_json`, and optional `metadata_json` / `preference_json`, matching TRL `reward_funcs` conventions while preserving backward compatibility with older manifests.

## RL Manifest Versus SFT Manifest

The SFT manifest can include broader supervised examples. The RL manifest is a reward-verifiable subset with these fields per row:

- `sample_id`
- `source_dataset`
- `task_type`
- `split`
- `images`
- `messages`
- `target`
- `verifier`
- `reward_meta`
- optional `preference`

For V1 GRPO, the default RL build config keeps a conservative single-image subset with `max_images_per_sample: 1`. SFT can still use max-3-image Phi-4 data through the active SFT split config.

The RL audit and validation scripts check image paths, prompt content, target/verifier completeness, split consistency, verifier support, reward module applicability, accepted labels/answers, management keywords, forbidden claims, expected decisions, invalid labels, and duplicate samples before expensive training.

## Reward Diagnostics

Reward-only diagnostics can be produced without training:

```bash
PYTHONPATH=src python3 scripts/score_rl_manifest.py \
  --manifest data/manifests/full/rl_manifest.jsonl \
  --output reports/rl_reward_report.jsonl \
  --summary-output reports/rl_reward_summary.json \
  --max-samples 200
```

The report includes per-module rewards, total reward, component nonzero counts,
hallucination penalty counts, uncertainty reward counts, zero/negative reward
counts, and a small reward histogram. GRPO runs can also write reward diagnostic
JSONL rows by setting `AGRI_VLM_REWARD_DIAGNOSTICS_JSONL`; the tiny Turin smoke
script sets this automatically.

## Current Training Gate

The SFT stage is still running. This task prepares code, configs, manifest validation, reward-only scoring, reward diagnostics, preference-data scaffolds, Slurm smoke wiring, and CPU-safe tests only. Formal RLFT must wait until a completed Phi-4 SFT checkpoint or adapter path exists under the SFT output area.

Future RLFT targets 4 NVIDIA B200 GPUs on one node with bf16 and `torchrun --nproc_per_node=4`. The default Slurm wrapper points to the smoke-after-SFT config, not the full formal config.

An optional smaller hpg-turin smoke script is available:

```bash
sbatch \
  --export=ALL,SFT_CHECKPOINT_PATH=/path/to/completed/sft/checkpoint_or_adapter \
  scripts/hpc/run_rl_grpo_phi4_turin8_tiny_smoke.slurm
```

It requests `hpg-turin`, 8 GPUs, 96 GB total RAM, and 45 minutes. It uses
`configs/train/rl_grpo_phi4_turin8_tiny_smoke.yaml`, 8 manifest samples, and 2
GRPO steps. It is only a model-load/reward/schema smoke test, not a formal RL
run.

## Known Limitations

- Rewards are heuristic and can still be gamed.
- Management coverage is still keyword based, even though repetition and list stuffing are now capped or penalized.
- Hallucination detection is still rule-based and depends on manifest metadata for visual evidence, crop, disease, allowed claims, and unsafe recommendations.
- Clarify detection is deterministic and imperfect.
- Default RL is single-image while active SFT uses max-3-image samples.
- No learned reward model is trained.
- Human preference pair data can be represented and exported, but it is not used by default and no reward model is trained.
- Post-RL evaluation still needs a complete before/after benchmark run.

## Next Steps Before Real GRPO

- Finish and freeze the SFT checkpoint or adapter path.
- Run full manifest validation and reward-only scoring on the selected RL train and holdout manifests.
- Review reward diagnostics for zero/negative reward spikes, overactive penalties, and missing component coverage.
- Run only the tiny GRPO smoke script first; inspect logs for model loading, reward diagnostics, schema failures, and non-finite rewards.
- Collect expert preference rows before adding any learned reward model.
- Only after those gates pass, prepare a formal GRPO run with explicit checkpoint, memory, and evaluation settings.

## Post-RL Evaluation Plan

Compare the completed Phi-4 SFT checkpoint before RLFT with the Phi-4 SFT plus GRPO checkpoint after RLFT.

Metrics to report:

- classification accuracy
- exact match / acceptable-answer accuracy
- clarify accuracy
- clarify precision
- clarify recall
- unnecessary clarification rate
- premature answer rate
- structured consultation compliance
- management coverage proxy
- hallucination / overconfidence proxy
- average composite reward
- task-wise breakdown

Use the phase-aware benchmark harness for the formal comparison:

```bash
PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase rl \
  --split val \
  --model-key agvlm_phi4_sft_completed \
  --max-samples 2 \
  --dry-run

PYTHONPATH=benchmarks/vlm_baselines python3 benchmarks/vlm_baselines/run_baselines.py \
  --phase rl \
  --split val \
  --model-key agvlm_phi4_rl_completed \
  --max-samples 2 \
  --dry-run
```

The full RL benchmark should compare external baselines, the completed SFT checkpoint, and the completed RL checkpoint on the same `rl_benchmark` split. Reward scores, if exported, are diagnostics only and must be reported separately from primary benchmark metrics.

Existing local and MIRAGE evaluation scripts cover parts of the classification, VQA, and holdout reporting path. Clarify precision/recall, structured compliance, management coverage, hallucination proxy, and average composite reward should be added or explicitly computed from prediction artifacts after RLFT.
