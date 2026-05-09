# RLFT Design

## Scope

RLFT in this repository means rule-based GRPO post-training for reward-verifiable agricultural VLM behavior. V1 remains scoped to ground-level RGB agricultural consultation, classification, VQA, management, uncertainty, and clarify-vs-respond behavior.

This is not full RLHF. The current implementation does not train or load a learned reward model, and it does not implement PPO. Rewards are deterministic or semi-deterministic functions over the model completion plus manifest verifier fields.

## Why GRPO

The target behaviors have verifiable pieces: labels, accepted answers, required structured sections, clarification decisions, uncertainty flags, management keywords, and forbidden claims. TRL `GRPOTrainer` supports custom `reward_funcs`, so the repo can compute one scalar reward per completion without preference pairs or a learned reward model.

GRPO is used only after SFT. Non-dry-run GRPO requires `sft_checkpoint_path` to point to an existing completed SFT checkpoint or adapter. The trainer rejects placeholders, missing paths, and the raw base model name `microsoft/Phi-4-reasoning-vision-15B`.

## Reward Modules

- `exact_match`: rewards exact normalized matches against target answers or accepted answers.
- `normalized_label`: rewards canonical agricultural label matches after label normalization.
- `synonym_match`: rewards acceptable synonym groups when exact labels differ.
- `structured_format`: rewards required consultation section headers such as Diagnosis, Evidence, Uncertainty, Management, and Follow-up.
- `uncertainty_calibration`: rewards uncertainty language when the verifier marks evidence as insufficient.
- `clarify_vs_respond`: rewards the correct high-level decision. JSON `{"decision": "clarify"}` or `{"decision": "respond"}` is honored when present; plain clarification questions are detected only when they are not substantive answers.
- `management_coverage`: rewards coverage of expected management keywords or steps.
- `hallucination_penalty`: penalizes configured forbidden claims and overconfident language when uncertainty is required.

The composite reward function is exposed through `make_trl_reward_function()` and accepts `prompts`, `completions`, `task_type`, `target_json`, `verifier_json`, and `reward_meta_json`, matching TRL `reward_funcs` conventions.

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

For V1 GRPO, the default RL build config keeps a conservative single-image subset with `max_images_per_sample: 1`. SFT can still use max-3-image Phi-4 data through the active SFT split config.

The RL audit script validates image paths, prompt content, target/verifier completeness, split exclusion, verifier support, reward module applicability, and single-image scope before expensive training.

## Current Training Gate

The SFT stage is still running. This task prepares code, configs, audit tools, reward sanity checks, Slurm wiring, and CPU-safe tests only. Formal RLFT must wait until a completed Phi-4 SFT checkpoint or adapter path exists under the SFT output area.

Future RLFT targets 4 NVIDIA B200 GPUs on one node with bf16 and `torchrun --nproc_per_node=4`. The default Slurm wrapper points to the smoke-after-SFT config, not the full formal config.

## Known Limitations

- Rewards are heuristic and can be gamed.
- Management coverage is keyword based.
- Hallucination detection is limited to configured forbidden claims and overconfidence markers.
- Clarify detection is deterministic and imperfect.
- Default RL is single-image while active SFT uses max-3-image samples.
- No learned reward model is trained.
- No human preference pair data is used.
- Post-RL evaluation still needs a complete before/after benchmark run.

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

Existing local and MIRAGE evaluation scripts cover parts of the classification, VQA, and holdout reporting path. Clarify precision/recall, structured compliance, management coverage, hallucination proxy, and average composite reward should be added or explicitly computed from prediction artifacts after RLFT.
