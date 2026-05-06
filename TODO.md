# TODO

## P0 Critical

- Submit the 100-step Llama 4 Scout B200 max3 probe after HPG maintenance.
  - Action: run `sbatch --export=ALL,TRAIN_CONFIG=configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm`.
  - Files: `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml`, `scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm`
  - Evidence: the retained upstream adapter exists at `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft`.
  - Success check: no OOM, stable step time, loss metrics written, checkpoint write succeeds at step 50 or 100.

- Launch the full Llama 4 Scout B200 max3 SFT run if the probe is healthy.
  - Action: run `sbatch scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm`.
  - Files: `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced.yaml`, `configs/data/sft_train_eval_llama4_max3.yaml`, `configs/deepspeed/zero3_lora_b200_no_offload.json`
  - Rationale: this is the next publishable SFT checkpoint path; do not resume from the failed AGBASE-disjoint continuation.

- Run post-SFT benchmark on the same local holdout and MIRAGE splits.
  - Action: after a completed full checkpoint exists, rerun `scripts/eval/run_benchmark.py` with `--checkpoint-path` against `local_holdout`, `mirage_mmst`, and `mirage_mmmt`.
  - Files: `scripts/eval/run_benchmark.py`, `configs/eval/local_holdout_full.yaml`, `configs/eval/mirage_mmst_full.yaml`, `configs/eval/mirage_mmmt_full.yaml`
  - Rationale: the before/after comparison requested by the project depends on matching eval conditions pre- and post-fine-tuning.

- Export SFT training artifacts once the full run completes.
  - Action: run `PYTHONPATH=src python scripts/artifacts/export_training_artifacts.py --run-dir <sft_run_dir>`.
  - Files: `scripts/artifacts/export_training_artifacts.py`, `docs/results_artifacts.md`
  - Rationale: paper figures should be regenerated from raw metrics, not manually recreated.

## P1 Important

- Keep generation evaluation out of large training jobs.
  - Action: leave `eval_generation_metrics: false` in full SFT configs and run generation evaluation separately on selected checkpoints.
  - Files: `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced.yaml`, `src/agri_vlm/evaluation/inference.py`
  - Rationale: the May 6 AGBASE-disjoint job stalled after step-500 loss eval because inline distributed generation metrics were too expensive.

- Review AGBASE target formatting before any future AGBASE-only or disjoint continuation.
  - Action: inspect AGBASE rows and validation predictions for free-form target drift; decide whether to normalize labels, tighten prompts, or keep AGBASE only inside balanced mixtures.
  - Files: `src/agri_vlm/data/normalizers.py`, `src/agri_vlm/data/conversation_format.py`, `outputs/benchmarks/` after future eval runs
  - Rationale: the disjoint continuation degraded validation quality and should not be used as the next-stage base without data/target review.

- Decide whether the evaluator should score normalized labels only or support free-form diagnoses.
  - Action: review future prediction JSONL outputs, then either tighten prompts or broaden the metric normalization.
  - Files: `src/agri_vlm/evaluation/local_eval.py`, `src/agri_vlm/evaluation/metrics.py`, `src/agri_vlm/data/conversation_format.py`
  - Rationale: model inference can be valid even when exact/normalized scoring is too strict for verbose agricultural answers.

- Improve PlantDoc multi-label handling.
  - Action: replace the current "most frequent category per image" heuristic with a better deterministic policy or multi-target representation after reviewing the official annotation distribution.
  - Files: `src/agri_vlm/data/hf_download.py`, `src/agri_vlm/data/normalizers.py`
  - Rationale: the current mapping is explicit and usable, but it compresses multi-object annotations into one label.

- Validate `flash-attn` against the CUDA 12.9.1 HiPerGator image.
  - Action: install with `INSTALL_FLASH_ATTN=1`, run `scripts/verify_environment.py`, and confirm at least one real SFT launch on B200 hardware.
  - Files: `scripts/hpc/prepare_env.sh`, `scripts/bootstrap_env.sh`, `README.md`
  - Rationale: the repo keeps `flash-attn` optional until the target image is confirmed.

## P2 Nice-to-Have

- Integrate AgMMU and AgroBench evaluators after access verification.
  - Action: verify official sources/licenses, add normalizers, add eval configs, and register tasks in `scripts/eval/run_benchmark.py`.
  - Files: `configs/benchmarks/benchmarks.yaml`, `scripts/benchmarks/benchmark_status.py`, `docs/benchmark_plan.md`
  - Rationale: they are important for the full paper matrix but should not block the minimum publishable pipeline.

- Add a dedicated `make data-smoke` target.
  - Action: expose the synthetic raw-data pipeline used in tests as a top-level Make target.
  - Files: `Makefile`, `scripts/data/prepare_manual_dataset_slots.py`, `scripts/data/normalize_all.py`
  - Rationale: the repo already has the pieces; a named target would make local validation easier.

- Add measured HiPerGator cache and scratch recommendations.
  - Action: record stable values for `HF_HOME`, `TMPDIR`, and dataset scratch usage after real cluster runs.
  - Files: `README.md`, `docs/decision_log.md`
  - Rationale: the current environment guidance is correct but not yet tuned with real cluster usage data.
