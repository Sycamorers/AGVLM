# Session Handoff

Last updated: 2026-05-06

## Current Active Milestone

Prepare and submit the Llama 4 Scout full-data max3 SFT run on 4x B200 after HPG maintenance.

Start with the 100-step probe:

```bash
cd /blue/hmedeiros/qinruoyao/agvlm

sbatch \
  --export=ALL,TRAIN_CONFIG=configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml \
  scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm
```

If the probe succeeds with no OOM, acceptable step time, and valid checkpoint writes, launch the full run:

```bash
sbatch scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm
```

## Artifact State

Keep this completed adapter on Orange:

```text
/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft
```

This is the upstream `sft_checkpoint_path` for the next-stage max3 run.

Local cleanup is intentional:

- `outputs/` now only contains the skeleton `outputs/sft/`.
- `logs/` now only contains the skeleton `logs/slurm/`.
- Orange SFT storage was reduced to the retained balanced adapter, about `87M`.

## May 6 Debug Summary

The AGBASE-only disjoint continuation was stopped at job `31951103`.

Observed state:

- reached `global_step=500`
- wrote step-500 loss eval
- did not write `checkpoint-500`
- last complete checkpoint before cleanup was `checkpoint-450`
- step-250 generation wrote `validation_predictions/step-250.jsonl` only after several hours
- step-500 repeated the same expensive generation-eval path

Conclusion:

- This was not a scheduler kill or missing checkpoint-output path.
- The job remained active on 4x B200s but was trapped in distributed generation metrics.
- Inline generation eval with the training-wrapped ZeRO-3 model is too expensive for long SFT jobs.
- The disjoint continuation degraded validation quality and should not be used as the next-stage base.

Quality comparison:

- retained balanced B200 run: step `2500`, eval loss `0.2343`, average reward `0.7480`
- disjoint clean restart: step `500`, eval loss `3.9104`; step-250 generation average reward `0.2871`

Operating rule:

- keep `eval_generation_metrics: false` in large SFT configs
- run generation evaluation separately on selected checkpoints

## Next-Stage Files

- `configs/data/sft_train_eval_llama4_max3.yaml`
- `configs/deepspeed/zero3_lora_b200_no_offload.json`
- `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml`
- `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced.yaml`
- `scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm`

The Slurm wrapper rebuilds:

- `data/manifests/full/sft_train_llama4_max3_no_eval_overlap.jsonl`
- `data/manifests/full/sft_eval_llama4_max3_stratified512.jsonl`
- `data/manifests/full/sft_train_eval_llama4_max3_summary.json`

## Monitor Commands

Probe:

```bash
tail -f logs/slurm/agri-vlm-sft-full-max3-b200-<job_id>.out
tail -f logs/slurm/agri-vlm-sft-full-max3-b200-<job_id>.err
tail -n 40 outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced-probe/metrics.jsonl
ls -lh /orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced-probe/
```

Full run:

```bash
tail -f logs/slurm/agri-vlm-sft-full-max3-b200-<job_id>.out
tail -f logs/slurm/agri-vlm-sft-full-max3-b200-<job_id>.err
tail -n 40 outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced/metrics.jsonl
ls -lh /orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced/
```

TensorBoard:

```bash
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

tensorboard \
  --logdir outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced/tensorboard \
  --host 0.0.0.0 \
  --port 6013
```

## After Full SFT Completes

1. Export training artifacts:

```bash
PYTHONPATH=src python scripts/artifacts/export_training_artifacts.py \
  --run-dir outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced
```

2. Run local holdout and MIRAGE benchmarks:

```bash
PYTHONPATH=src python scripts/eval/run_benchmark.py \
  --model-config configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml \
  --tasks local_holdout mirage_mmst mirage_mmmt \
  --prediction-mode model \
  --checkpoint-path <checkpoint_or_adapter_dir> \
  --output-dir outputs/benchmarks/llama4-scout-full-max3
```

3. Export benchmark tables.
4. Decide whether the full max3 checkpoint is strong enough to seed GRPO.

## Historical Notes

- April L4/Qwen SFT hit CUDA OOM during fp32 loss conversion. The chunked-loss mitigation remains in `src/agri_vlm/training/sft_trainer.py`, but the L4 rerun is not the active milestone.
- Earlier baseline Qwen local-holdout inference worked operationally but scored all zero under exact/normalized metrics because verbose free-text answers did not match the evaluator assumptions.
