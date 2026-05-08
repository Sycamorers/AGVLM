# Progress Tracker

Current active milestone: Phi-4 reasoning vision full-data SFT on the
max-3-image manifest using 16 Turin L4 GPUs.

| Area | Status | Notes |
| --- | --- | --- |
| Data split | ready | `configs/data/sft_train_eval_phi4_max3.yaml` rebuilds non-overlapping train/eval manifests. |
| Model config | ready | `configs/model/phi4_reasoning_vision_15b_turin_24g.yaml` uses Phi-4 reasoning vision processing. |
| Batch preflight | ready | Slurm wrapper tests `PHI4_BATCH_CANDIDATES` before full training. |
| Full SFT | pending | Submit `scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm`. |
| Post-SFT eval | blocked | Waiting for a completed SFT checkpoint. |
| GRPO | blocked | Waiting for SFT and post-SFT evaluation. |
