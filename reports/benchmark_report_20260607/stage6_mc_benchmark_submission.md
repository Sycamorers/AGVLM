# Stage6 MC Benchmark Submission

Date: 2026-06-07

## Decision

The next required step is to benchmark the Stage6 multiple-choice SFT candidate on the same Stage5 held-out benchmark split used for the latest completed benchmark report.

The explicit `checkpoint-160` directory is a DeepSpeed checkpoint directory and contains a placeholder-sized `adapter_model.safetensors`. The usable final PEFT adapter is the run root:

`/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-classification-probe-stage6-mc-b200-4gpu`

That root adapter validates as a PEFT LoRA adapter with 320 non-empty tensors and is the path configured by `agvlm_phi4_sft_classification_probe_stage6_mc_b200_candidate`.

## Validation

Dry-run command:

```bash
PYTHONPATH=$PWD/benchmarks/vlm_baselines \
/blue/hmedeiros/qinruoyao/.conda/envs/agri-vlm-v1/bin/python \
  benchmarks/vlm_baselines/run_baselines.py \
  --phase sft \
  --split test \
  --split-dir benchmarks/vlm_baselines/splits_stage5_datafix \
  --model-key agvlm_phi4_sft_classification_probe_stage6_mc_b200_candidate \
  --output-dir benchmarks/vlm_baselines/results/agvlm_stage6_mc_benchmark_20260607 \
  --dtype bf16 \
  --dry-run
```

Dry run selected 736 test samples from `benchmarks/vlm_baselines/splits_stage5_datafix/sft_test_manifest.jsonl` and resolved 4-bit inference.

## Submitted Job

Slurm job: `34069292`

```bash
sbatch \
  --job-name=agri-sft-bench-stage6-mc \
  --export=ALL,MODEL_KEY=agvlm_phi4_sft_classification_probe_stage6_mc_b200_candidate,SPLIT=test,SPLIT_DIR=benchmarks/vlm_baselines/splits_stage5_datafix,OUTPUT_DIR=benchmarks/vlm_baselines/results/agvlm_stage6_mc_benchmark_20260607,DTYPE=bf16,MIN_NEW_TOKENS=2,MAX_NEW_TOKENS=0,MAX_SAMPLES=0 \
  benchmarks/vlm_baselines/slurm/run_sft_benchmark_24gb.sbatch
```

Expected outputs:

- `benchmarks/vlm_baselines/results/agvlm_stage6_mc_benchmark_20260607/predictions/`
- `benchmarks/vlm_baselines/results/agvlm_stage6_mc_benchmark_20260607/metrics/summary_table.csv`
- `benchmarks/vlm_baselines/results/agvlm_stage6_mc_benchmark_20260607/metadata/`
