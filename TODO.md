# TODO

## Active

- Monitor the submitted Phi-4 reasoning vision Turin 16-GPU SFT job.
  - Slurm wrapper: `scripts/hpc/run_sft_turin_16gpu_phi4_reasoning_vision_15b_full_max3.slurm`
  - Model config: `configs/model/phi4_reasoning_vision_15b_turin_24g.yaml`
  - Train config: `configs/train/sft_phi4_reasoning_vision_15b_turin_16gpu_full_max3.yaml`
  - Data config: `configs/data/sft_train_eval_phi4_max3.yaml`
  - Success check: batch-size preflight selects the largest passing candidate, full training starts, metrics are written, and checkpoint writes succeed.

- After the full SFT checkpoint is available, run local holdout and MIRAGE
  benchmarks against the selected Phi-4 run directory.

- Keep inline generation metrics disabled during long SFT jobs; run generation
  evaluation separately on selected checkpoints.
