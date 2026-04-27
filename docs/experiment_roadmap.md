# Experiment Roadmap

## Current Active Milestone

Full-data SFT on L4 has been submitted. The latest visible local Slurm log is `logs/slurm/agri-vlm-sft-full-l4-30580348.err`, which shows a CUDA OOM on April 21, 2026 during the default Qwen3-VL loss path when logits were cast to fp32.

The repo currently contains a mitigation in progress: `loss_chunk_size: 1024` in the L4 configs and a chunked SFT loss implementation. Treat the active milestone as "full SFT pending rerun or resume" until a completed checkpoint exists under `outputs/sft/qwen3-vl-4b-lora-full-l4/`.

## Phase Gates

| Phase | Objective | Exit Criteria | Status |
| --- | --- | --- | --- |
| Data readiness | Build full SFT, RL, and eval manifests | `data/manifests/full/` reports available | completed |
| Base benchmark | Confirm evaluation wiring on base model | benchmark summary under `outputs/benchmarks/` | completed for local slice |
| SFT | Produce agricultural SFT checkpoint | final checkpoint, metrics, TensorBoard logs | running or pending rerun |
| Post-SFT eval | Compare base vs SFT | benchmark summary and tables | next |
| GRPO | Optimize clarify-vs-respond policy | GRPO checkpoint with reward curves | blocked by SFT checkpoint |
| Final eval | Run benchmark matrix | tables, figures, error summaries | blocked by SFT and GRPO |
| Ablations | Validate method contributions | no-RL, reward, LoRA, freeze, data mixture reports | optional after main result |

## SFT Decision Tree

If SFT succeeds:

1. Export training curves from the SFT run.
2. Run local holdout and MIRAGE benchmarks against the SFT checkpoint.
3. Export benchmark tables.
4. Start GRPO from the SFT checkpoint.

If SFT fails with CUDA OOM:

1. Confirm the rerun used `loss_chunk_size: 1024`.
2. Reduce `gradient_accumulation_steps` only if per-step activation memory, not effective batch size, is the issue.
3. Reduce image resolution through the model config `max_pixels`.
4. Prefer B200 SFT config if L4 remains memory-bound.
5. Keep the failed run logs under `logs/slurm/` and record the failure in `docs/progress_tracker.md`.

If SFT fails before training starts:

1. Run `PYTHONPATH=src python scripts/verify_environment.py`.
2. Run SFT with `--dry-run`.
3. Validate `data/manifests/full/sft_manifest.valid_images.jsonl`.
4. Check distributed variables and Slurm node allocation.

## Ablation Tracks

- No-RL: Agri-SFT vs Agri-SFT + GRPO.
- No clarify-aware construction: generic SFT vs Agri-SFT.
- Reward component ablation: remove `clarify_vs_respond`, `uncertainty_calibration`, `hallucination_penalty`, or `management_coverage`.
- LoRA scope: attention-only vs attention + MLP.
- Freeze strategy: freeze vision and train projector; freeze vision plus freeze projector; optional partial visual unfreezing.
- Data mixture: diagnosis-only, management-only, consultation-only, full mixture.
- Ambiguity analysis: low, medium, and high ambiguity subsets.
