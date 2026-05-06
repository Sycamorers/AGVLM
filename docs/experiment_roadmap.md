# Experiment Roadmap

## Current Active Milestone

The active milestone is Llama 4 Scout full-data SFT on the max-3-image manifest using 4x B200 GPUs. The next command after HPG maintenance is the 100-step probe from the retained balanced adapter:

```bash
sbatch \
  --export=ALL,TRAIN_CONFIG=configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml \
  scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm
```

If the probe is healthy, submit:

```bash
sbatch scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm
```

The retained adapter is:

```text
/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft
```

The AGBASE-disjoint continuation from job `31951103` is excluded from the next-stage path. It reached `global_step=500` but stalled during inline distributed generation evaluation and did not write `checkpoint-500`; the degraded checkpoint paths were removed.

## Phase Gates

| Phase | Objective | Exit Criteria | Status |
| --- | --- | --- | --- |
| Data readiness | Build full SFT, RL, and eval manifests | `data/manifests/full/` reports available | completed |
| Max3 split | Build no-overlap Llama 4 max3 train/eval manifests | summary JSON and train/eval JSONL exist | ready at launch |
| Base benchmark | Confirm evaluation wiring on base or retained adapter | benchmark summary under `outputs/benchmarks/` | completed for earlier local slice; rerun needed for final model family |
| SFT probe | Validate full max3 B200 memory and checkpoint writes | 100-step probe completes | next |
| Full SFT | Produce agricultural full max3 SFT checkpoint | final checkpoint, metrics, TensorBoard logs | blocked by probe |
| Post-SFT eval | Compare base or retained adapter vs full max3 SFT | benchmark summary and tables | blocked by SFT checkpoint |
| GRPO | Optimize clarify-vs-respond policy | GRPO checkpoint with reward curves | blocked by SFT checkpoint and eval |
| Final eval | Run benchmark matrix | tables, figures, error summaries | blocked by SFT and GRPO |
| Ablations | Validate method contributions | no-RL, reward, LoRA, freeze, data mixture reports | optional after main result |

## SFT Decision Tree

If the 100-step probe succeeds:

1. Confirm checkpoint files exist in the probe Orange output directory.
2. Inspect loss trend, step time, and Slurm warnings.
3. Launch the full max3 B200 run.

If the probe fails with CUDA OOM:

1. Confirm the run used `configs/model/llama4_scout_17b_16e_turin_24g_lowres.yaml`.
2. Lower image resolution through the model config `max_pixels`.
3. Consider smaller `loss_chunk_size` or dataloader worker count before changing the data mixture.
4. Keep the failed run logs under `logs/slurm/` and record the job ID in `docs/progress_tracker.md`.

If the probe fails before training starts:

1. Run `PYTHONPATH=src python scripts/verify_environment.py`.
2. Verify Hugging Face access to `meta-llama/Llama-4-Scout-17B-16E-Instruct`.
3. Rebuild `data/manifests/full/sft_train_llama4_max3_no_eval_overlap.jsonl`.
4. Check Orange checkpoint permissions and quota.

If full SFT succeeds:

1. Export training curves from the full SFT run.
2. Run local holdout and MIRAGE benchmarks against the completed checkpoint.
3. Export benchmark tables.
4. Start GRPO from the selected SFT checkpoint.

## Historical Debug Notes

- The April L4 Qwen path hit CUDA OOM during fp32 loss conversion; chunked loss remains useful code, but this is no longer the active milestone.
- The May 6 B200 AGBASE-disjoint continuation froze operationally because inline generation metrics were too expensive in the distributed training job. Future large SFT configs must keep generation metrics disabled and run generation evaluation separately.

## Ablation Tracks

- No-RL: Agri-SFT vs Agri-SFT + GRPO.
- No clarify-aware construction: generic SFT vs Agri-SFT.
- Reward component ablation: remove `clarify_vs_respond`, `uncertainty_calibration`, `hallucination_penalty`, or `management_coverage`.
- LoRA scope: attention-only vs attention + MLP.
- Freeze strategy: freeze vision and train projector; freeze vision plus freeze projector; optional partial visual unfreezing.
- Data mixture: diagnosis-only, management-only, consultation-only, full mixture.
- Ambiguity analysis: low, medium, and high ambiguity subsets.
