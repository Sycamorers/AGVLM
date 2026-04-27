# TODO

## P0 Critical

- Rerun or resume full SFT after the L4 OOM.
  - Action: submit the L4 full SFT config with `loss_chunk_size: 1024`, or use the B200 full config if L4 remains memory-bound.
  - Files: `configs/train/sft_lora_full_l4_multigpu.yaml`, `scripts/hpc/run_sft_full_l4.slurm`, `src/agri_vlm/training/sft_trainer.py`
  - Evidence: `logs/slurm/agri-vlm-sft-full-l4-30580348.err` failed with CUDA OOM during fp32 loss conversion.
  - Rationale: the first full Agri-SFT checkpoint blocks post-SFT eval and GRPO.

- Run post-SFT benchmark on the same local holdout split.
  - Action: after SFT writes a completed checkpoint, rerun `scripts/eval/run_benchmark.py` with `--checkpoint-path` against `local_holdout`, `mirage_mmst`, and `mirage_mmmt`.
  - Files: `scripts/eval/run_benchmark.py`, `configs/eval/local_holdout_full.yaml`, `configs/eval/mirage_mmst_full.yaml`, `configs/eval/mirage_mmmt_full.yaml`
  - Rationale: the before/after comparison requested by the project depends on matching eval conditions pre- and post-fine-tuning.

- Export SFT training artifacts once the run completes.
  - Action: run `PYTHONPATH=src python scripts/artifacts/export_training_artifacts.py --run-dir <sft_run_dir>`.
  - Files: `scripts/artifacts/export_training_artifacts.py`, `docs/results_artifacts.md`
  - Rationale: paper figures should be regenerated from raw metrics, not manually recreated.

## P1 Important

- Decide whether the evaluator should score normalized labels only or support free-form diagnoses.
  - Action: review `outputs/benchmarks/base-qwen3-vl-4b_local_holdout_256/local_holdout/predictions.jsonl`, then either tighten prompts or broaden the metric normalization.
  - Files: `src/agri_vlm/evaluation/local_eval.py`, `src/agri_vlm/evaluation/metrics.py`, `src/agri_vlm/data/conversation_format.py`
  - Rationale: the base model inference path now works, but the current metric path gives all-zero scores because the model answers verbosely.

- Improve PlantDoc multi-label handling.
  - Action: replace the current “most frequent category per image” heuristic with a better deterministic policy or multi-target representation after reviewing the official annotation distribution.
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
