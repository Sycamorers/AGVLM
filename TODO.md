# TODO

## P0 Critical

- Validate AgMMU automatic partial download on HiPerGator.
  - Action: run `scripts/data/download_public_datasets.py --download-mode partial --fraction 0.1 --datasets agmmu` inside the Python 3.11 conda env, confirm that the first 10% of the train split is materialized without a full archive download, then normalize and build the AgMMU eval manifest.
  - Files: `src/agri_vlm/data/hf_download.py`, `scripts/data/download_public_datasets.py`, `scripts/data/normalize_agmmu.py`
  - Rationale: the adapter is implemented against the public Hugging Face dataset, but it still needs a real cluster validation run.

- Validate AgroBench gated download with a real authenticated Hugging Face session.
  - Action: request access to `risashinoda/AgroBench`, authenticate on HiPerGator, rerun the partial download path, and confirm the eval manifest is produced without leaking benchmark-only data into training manifests.
  - Files: `configs/data/datasets.yaml`, `src/agri_vlm/data/hf_download.py`, `scripts/data/normalize_agrobench.py`
  - Rationale: AgroBench is now first-class in the registry, but the public environment used for this pass cannot verify gated access.

- Stage manual subset-tagged raw data for IP102, AgBase resources, and Agri-LLaVA.
  - Action: place approved raw data under `data/raw/<dataset>/<subset_tag>/`, rerun `scripts/data/normalize_all.py`, and confirm the merged SFT and RL manifests include the expected rows.
  - Files: `configs/data/datasets.yaml`, `scripts/data/prepare_manual_dataset_slots.py`, `scripts/data/normalize_all.py`
  - Rationale: these sources do not have a verified selective remote-download path, so the repo now makes the missing manual step explicit.

- Run one real partial-to-full rerun on HiPerGator.
  - Action: execute the full data workflow once with `partial/0.1` and once with `full/1.0`, then compare row counts and verify that raw, interim, manifest, and report outputs remain isolated by subset tag.
  - Files: `scripts/data/*.py`, `scripts/hpc/run_data_prep.sh`, `scripts/hpc/run_data_prep.slurm`
  - Rationale: the code supports `partial_10pct` and `full`, but the upgrade still needs an end-to-end cluster validation run.

## P1 Important

- Improve PlantDoc multi-label handling.
  - Action: replace the current “most frequent category per image” heuristic with a better deterministic policy or multi-target representation after reviewing the official annotation distribution.
  - Files: `src/agri_vlm/data/hf_download.py`, `src/agri_vlm/data/normalizers.py`
  - Rationale: the current mapping is explicit and usable, but it compresses multi-object annotations into one label.

- Add a small integration test for authenticated and gated dataset failure modes.
  - Action: mock gated Hugging Face failures and assert that manual slots, placeholder manifests, and report statuses are emitted correctly.
  - Files: `src/agri_vlm/data/hf_download.py`, `tests/`
  - Rationale: manual fallback behavior is important and currently covered only indirectly.

- Validate `flash-attn` against the CUDA 12.9.1 HiPerGator image.
  - Action: install with `INSTALL_FLASH_ATTN=1`, run `scripts/verify_environment.py`, and confirm at least one real SFT launch on B200 hardware.
  - Files: `scripts/hpc/prepare_env.sh`, `scripts/bootstrap_env.sh`, `README.md`
  - Rationale: the repo keeps `flash-attn` optional until the target image is confirmed.

## P2 Nice-to-Have

- Add a dedicated `make data-smoke` target.
  - Action: expose the synthetic raw-data pipeline used in tests as a top-level Make target.
  - Files: `Makefile`, `scripts/data/prepare_manual_dataset_slots.py`, `scripts/data/normalize_all.py`
  - Rationale: the repo already has the pieces; a named target would make local validation easier.

- Add measured HiPerGator cache and scratch recommendations.
  - Action: record stable values for `HF_HOME`, `TMPDIR`, and dataset scratch usage after real cluster runs.
  - Files: `README.md`, `docs/decision_log.md`
  - Rationale: the current environment guidance is correct but not yet tuned with real cluster usage data.
