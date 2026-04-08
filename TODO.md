# TODO

## P0 Critical

- Validate `flash-attn` build against CUDA 12.9.1 on the target B200 image.
  - Action: install with `INSTALL_FLASH_ATTN=1`, run `scripts/verify_environment.py`, then complete at least one real SFT step and one RL step.
  - Files: `scripts/bootstrap_env.sh`, `README.md`, `pyproject.toml`
  - Rationale: the repo documents `flash-attn` as optional because the build has not been verified on the target environment yet.

- Run end-to-end multi-GPU checkpoint and resume validation on B200 hardware.
  - Action: launch `configs/train/sft_lora_b200_multigpu.yaml` and `configs/train/rl_grpo_b200_multigpu.yaml` with `scripts/launch_torchrun.py`, interrupt after at least one save, then resume from the saved checkpoint.
  - Files: `scripts/train/train_sft.py`, `scripts/train/train_rl_grpo.py`, `src/agri_vlm/training/`, `configs/train/`
  - Rationale: the distributed code path is wired and smoke-tested, but not yet exercised on real target hardware.

- Confirm the SFT checkpoint handoff into RL with a real adapter checkpoint.
  - Action: produce a real LoRA SFT checkpoint, then verify `load_sft_checkpoint_model()` resumes it correctly inside GRPO training.
  - Files: `src/agri_vlm/modeling/model_factory.py`, `src/agri_vlm/training/rl_trainer.py`
  - Rationale: the adapter-aware RL load path is implemented but still needs hardware validation against an actual saved checkpoint.

## P1 Important

- Add an integration test that launches `scripts/verify_environment.py` under `torchrun` when at least 2 GPUs are visible.
  - Action: gate the test on GPU availability and assert backend, world size, and local rank reporting.
  - Files: `tests/`, `scripts/verify_environment.py`
  - Rationale: current automated coverage only validates launcher command construction, not a live distributed process group.

- Add a containerized environment path for CUDA 12.9.1.
  - Action: add a `Dockerfile` and a short build/run note once the exact base image and optional `flash-attn` flow are validated.
  - Files: `Dockerfile`, `README.md`, `scripts/bootstrap_env.sh`
  - Rationale: the repo currently standardizes the local bootstrap path only.

- Improve rank-aware metrics persistence for RL runs.
  - Action: add a JSONL metrics sink for GRPO similar to SFT and verify only global rank 0 writes artifacts.
  - Files: `src/agri_vlm/training/rl_trainer.py`, `src/agri_vlm/training/callbacks.py`
  - Rationale: training logs are readable now, but RL artifact logging is still lighter than the SFT path.

## P2 Nice-to-Have

- Add an optional advanced distributed config path.
  - Action: add a clean DeepSpeed or FSDP example only after the primary `torchrun` path is validated on target hardware.
  - Files: `configs/`, `README.md`
  - Rationale: the repo should keep one primary distributed path until the basic B200 workflow is confirmed.

- Add a small benchmark note for recommended per-GPU batch sizes on 4B and 8B checkpoints.
  - Action: record memory observations and stable batch/accumulation settings after real B200 runs.
  - Files: `README.md`, `docs/decision_log.md`
  - Rationale: current B200 configs are sensible starting points, not measured performance baselines.
