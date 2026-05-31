# Classification Repair Stage2 B200 Progress

## Objective

Continue the classification-repair SFT from the completed Turin pilot because the pilot still produced blank `Answer:` classification outputs in the SFT benchmark. The stage2 run keeps V1 scoped to ground-level RGB agricultural consultation and classification tasks.

## Prompt and Data Changes

- Classification benchmark prompts now match the instructional SFT contract:
  - `Answer: <canonical agricultural label>`
  - `Evidence: <brief visible symptom evidence>`
  - `Do not leave Answer blank or copy the placeholder text.`
- Instructional classification SFT prompts use the same nonblank answer guard.
- Classification instructional SFT targets include `Answer:` plus `Evidence:`.
- Data prep uses `configs/data/sft_classification_repair_phi4_max3.yaml` and writes `data/manifests/full/sft_train_phi4_max3_classification_repair_instructional.jsonl`.

## Pre-Training Validation

- Full test suite passed before launching B200 training:
  - `/blue/hmedeiros/qinruoyao/.conda/envs/agri-vlm-v1/bin/python -m pytest`
  - Result: `143 passed, 2 warnings`
- New B200 train configs validated with `TrainConfigSchema`.
- Dry run passed for `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_classification_repair_instructional_stage2.yaml`.
- Data prep completed with `167710` rows:
  - classification: `86228`
  - VQA: `50000`
  - consultation: `25000`
  - clarify/respond: `6482`

## Pilot Benchmark Recheck

Prompt alignment alone did not fix the completed pilot:

- Slurm job: `33598510`
- Output: `benchmarks/vlm_baselines/results/agvlm_classification_repair_pilot_benchmark_promptfix_20260531`
- `task_macro_average`: `0.23425272331154684`
- `classification_macro_f1`: `0.0`
- Classification predictions remained blank `Answer:` outputs.

## B200 Stage2 Training

- Slurm job: `33598617`
- Partition/node: `hpg-b200`, `c1004a-s15`
- Allocation: `4` B200 GPUs, `800G` memory
- Wrapper: `scripts/hpc/run_sft_b200_4gpu_phi4_reasoning_vision_15b_full_max3.slurm`
- Preflight config: `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_classification_repair_instructional_preflight.yaml`
- Full config: `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_classification_repair_instructional_stage2.yaml`
- Full adapter path: `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-classification-repair-instructional-stage2-b200-4gpu`

Preflight completed successfully:

- `B200 preflight succeeded.`
- `train_loss`: `5.6258416175842285`
- `train_runtime`: `64.2387`

Full training started:

- `B200_FULL_TRAINING_STARTED=1`
- Initial full-training metrics through step 10: losses remained finite and the latest logged loss was `4.7112`
- First configured checkpoint: step `100`
- Evaluation: every `200` steps with loss-only validation

## Next Checks

1. Monitor job `33598617` through checkpoint `100` and final step `800`.
2. Validate the final adapter has non-empty LoRA tensors and an `adapter_validation.json`.
3. Benchmark model key `agvlm_phi4_sft_classification_repair_instructional_stage2_b200_candidate`.
4. Promote only if classification invalid predictions drop substantially from `114/114`, classification macro F1 rises above `0.0`, and task macro/clarify metrics do not regress against the active SFT gate.
