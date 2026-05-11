PYTHON ?= python3
NPROC_PER_NODE ?=
MASTER_PORT ?= 29500
DATA_DOWNLOAD_MODE ?= partial
DATA_FRACTION ?= 0.1
RL_MANIFEST ?= data/manifests/full/rl_manifest.jsonl
RL_EVAL_MANIFEST ?= data/manifests/full/rl_local_holdout_eval.jsonl
RL_PHI4_MODEL_CONFIG ?= configs/model/phi4_reasoning_vision_15b_turin_24g.yaml
RL_PHI4_READINESS_CONFIG ?= configs/train/rl_grpo_phi4_readiness.yaml
RL_PHI4_SMOKE_CONFIG ?= configs/train/rl_grpo_phi4_smoke.yaml
RL_PHI4_FULL_CONFIG ?= configs/train/rl_grpo_phi4_full.yaml
BENCHMARK_MODEL ?= HuggingFaceTB/SmolVLM2-2.2B-Instruct
BENCHMARK_MODEL_KEY ?= agvlm_phi4_sft_completed
BENCHMARK_RL_MODEL_KEY ?= agvlm_phi4_rl_completed
BENCHMARK_MAX_SAMPLES ?= 2

.PHONY: help bootstrap verify verify-dist smoke prepare-slots download-data normalize-all data-partial data-full data-report build-sft-manifest build-rl-manifest build-eval-manifest rl-data-full rl-audit-full rl-reward-check-full rl-format-check-full rl-phi4-readiness rl-phi4-smoke-after-sft rl-phi4-full-after-sft test-rl benchmark-status benchmark-phase-splits benchmark-sft-splits benchmark-rl-splits benchmark-dry-run benchmark-sft-dry-run benchmark-rl-dry-run benchmark-sft-summary benchmark-rl-summary benchmark-all-summary benchmark-sft-agvlm-checkpoint-dry-run benchmark-rl-agvlm-sft-checkpoint-dry-run benchmark-rl-agvlm-rl-checkpoint-dry-run benchmark-rl-agvlm-checkpoint-dry-run export-training-artifacts export-benchmark-tables sft sft-dist rl rl-dist eval eval-local test

help:
	@echo "Targets:"
	@echo "  bootstrap            Create a local virtualenv with Python 3.11"
	@echo "  verify               Verify runtime and package prerequisites"
	@echo "  verify-dist          Verify a distributed torchrun environment"
	@echo "  smoke                Run the smoke test pipeline"
	@echo "  prepare-slots        Create subset-tagged dataset slots and smoke raw data"
	@echo "  download-data        Download dataset subsets into raw storage"
	@echo "  normalize-all        Normalize all available datasets for the active subset tag"
	@echo "  data-partial         Download, normalize, build manifests, and report for the default 10 percent subset"
	@echo "  data-full            Download, normalize, build manifests, and report for the full dataset pass"
	@echo "  data-report          Generate the dataset report for the active subset tag"
	@echo "  build-sft-manifest   Build the SFT manifest"
	@echo "  build-rl-manifest    Build the RL manifest"
	@echo "  build-eval-manifest  Build evaluation manifests"
	@echo "  rl-data-full         Build the full reward-verifiable RL manifest"
	@echo "  rl-audit-full        Audit the full RL manifest"
	@echo "  rl-reward-check-full Run reward sanity checks on the full RL manifest"
	@echo "  rl-format-check-full Validate full RL dataset columns/prompts"
	@echo "  rl-phi4-readiness    Run Phi-4 GRPO readiness dry-run"
	@echo "  rl-phi4-smoke-after-sft  Print future smoke-after-SFT Slurm command after checkpoint validation"
	@echo "  rl-phi4-full-after-sft   Print future full 4x B200 Slurm command after checkpoint validation"
	@echo "  test-rl              Run RL-specific CPU-safe tests"
	@echo "  benchmark-status     Report benchmark readiness and missing paths"
	@echo "  benchmark-phase-splits Build SFT and RL benchmark split manifests"
	@echo "  benchmark-dry-run    Validate one external baseline on SFT and RL val splits without model load"
	@echo "  benchmark-sft-summary Refresh SFT/RL benchmark summary table from metrics files"
	@echo "  benchmark-sft-agvlm-checkpoint-dry-run Validate completed SFT checkpoint config without model load"
	@echo "  benchmark-rl-agvlm-rl-checkpoint-dry-run Validate completed RL checkpoint config without model load"
	@echo "  export-training-artifacts  Export curves/tables from RUN_DIR"
	@echo "  export-benchmark-tables    Export tables from BENCHMARK_RUNS"
	@echo "  sft                  Run 1-process supervised fine-tuning"
	@echo "  sft-dist             Run single-node multi-GPU supervised fine-tuning via torchrun"
	@echo "  rl                   Run 1-process GRPO post-training"
	@echo "  rl-dist              Run single-node multi-GPU GRPO post-training via torchrun"
	@echo "  eval                 Run local holdout evaluation"
	@echo "  test                 Run unit tests"

bootstrap:
	bash scripts/bootstrap_env.sh

verify:
	PYTHONPATH=src $(PYTHON) scripts/verify_environment.py

verify-dist:
	PYTHONPATH=src $(PYTHON) scripts/launch_torchrun.py \
		$(if $(NPROC_PER_NODE),--nproc-per-node $(NPROC_PER_NODE),) \
		--master-port $(MASTER_PORT) \
		scripts/verify_environment.py

prepare-slots:
	PYTHONPATH=src $(PYTHON) scripts/data/prepare_manual_dataset_slots.py \
		--download-mode $(DATA_DOWNLOAD_MODE) \
		--fraction $(DATA_FRACTION) \
		--with-smoke-data

download-data:
	PYTHONPATH=src $(PYTHON) scripts/data/download_public_datasets.py \
		--download-mode $(DATA_DOWNLOAD_MODE) \
		--fraction $(DATA_FRACTION)

normalize-all:
	PYTHONPATH=src $(PYTHON) scripts/data/normalize_all.py \
		--download-mode $(DATA_DOWNLOAD_MODE) \
		--fraction $(DATA_FRACTION)

data-partial:
	$(MAKE) download-data DATA_DOWNLOAD_MODE=partial DATA_FRACTION=0.1
	$(MAKE) normalize-all DATA_DOWNLOAD_MODE=partial DATA_FRACTION=0.1
	$(MAKE) build-sft-manifest DATA_DOWNLOAD_MODE=partial DATA_FRACTION=0.1
	$(MAKE) build-rl-manifest DATA_DOWNLOAD_MODE=partial DATA_FRACTION=0.1
	$(MAKE) build-eval-manifest DATA_DOWNLOAD_MODE=partial DATA_FRACTION=0.1
	$(MAKE) data-report DATA_DOWNLOAD_MODE=partial DATA_FRACTION=0.1

data-full:
	$(MAKE) download-data DATA_DOWNLOAD_MODE=full DATA_FRACTION=1.0
	$(MAKE) normalize-all DATA_DOWNLOAD_MODE=full DATA_FRACTION=1.0
	$(MAKE) build-sft-manifest DATA_DOWNLOAD_MODE=full DATA_FRACTION=1.0
	$(MAKE) build-rl-manifest DATA_DOWNLOAD_MODE=full DATA_FRACTION=1.0
	$(MAKE) build-eval-manifest DATA_DOWNLOAD_MODE=full DATA_FRACTION=1.0
	$(MAKE) data-report DATA_DOWNLOAD_MODE=full DATA_FRACTION=1.0

build-sft-manifest:
	PYTHONPATH=src $(PYTHON) scripts/data/build_sft_manifest.py \
		--config configs/data/sft_build.yaml \
		--download-mode $(DATA_DOWNLOAD_MODE) \
		--fraction $(DATA_FRACTION)

build-rl-manifest:
	PYTHONPATH=src $(PYTHON) scripts/data/build_rl_manifest.py \
		--config configs/data/rl_build.yaml \
		--download-mode $(DATA_DOWNLOAD_MODE) \
		--fraction $(DATA_FRACTION)

rl-data-full:
	PYTHONPATH=src $(PYTHON) scripts/data/prepare_rl_datasets.py \
		--subset full \
		--skip-unavailable \
		--write-report
	PYTHONPATH=src $(PYTHON) scripts/data/build_rl_manifest.py \
		--config configs/data/rl_build.yaml \
		--download-mode full \
		--fraction 1.0

rl-audit-full:
	PYTHONPATH=src $(PYTHON) scripts/data/audit_rl_manifest.py \
		--manifest-path $(RL_MANIFEST) \
		--eval-manifest-path $(RL_EVAL_MANIFEST) \
		--output-json outputs/rl/audit/full_rl_manifest_audit.json \
		--output-md outputs/rl/audit/full_rl_manifest_audit.md \
		--fail-on-critical

rl-reward-check-full:
	PYTHONPATH=src $(PYTHON) scripts/train/rl_reward_sanity_check.py \
		--manifest-path $(RL_MANIFEST) \
		--config $(RL_PHI4_READINESS_CONFIG) \
		--output-json outputs/rl/audit/full_rl_reward_sanity.json \
		--output-md outputs/rl/audit/full_rl_reward_sanity.md \
		--max-samples 200

rl-format-check-full:
	PYTHONPATH=src $(PYTHON) scripts/train/check_rl_dataset_format.py \
		--manifest-path $(RL_MANIFEST) \
		--model-config $(RL_PHI4_MODEL_CONFIG) \
		--max-samples 8 \
		--output-json outputs/rl/audit/rl_dataset_format_check.json \
		--output-md outputs/rl/audit/rl_dataset_format_check.md

rl-phi4-readiness:
	PYTHONPATH=src $(PYTHON) scripts/train/train_rl_grpo.py \
		--model-config $(RL_PHI4_MODEL_CONFIG) \
		--train-config $(RL_PHI4_READINESS_CONFIG) \
		--dry-run

rl-phi4-smoke-after-sft:
	PYTHONPATH=src $(PYTHON) -c 'from pathlib import Path; from agri_vlm.schemas.config_schema import ModelConfigSchema, RLTrainConfigSchema, load_config; from agri_vlm.training.rl_trainer import validate_rl_sft_checkpoint_path; model=load_config(Path("$(RL_PHI4_MODEL_CONFIG)"), ModelConfigSchema); train=load_config(Path("$(RL_PHI4_SMOKE_CONFIG)"), RLTrainConfigSchema); validate_rl_sft_checkpoint_path(model, train); manifest=Path(train.manifest_path); assert manifest.exists(), "RL manifest missing: %s" % manifest'
	@echo "sbatch scripts/hpc/run_rl_grpo_phi4_smoke.slurm"

rl-phi4-full-after-sft:
	PYTHONPATH=src $(PYTHON) -c 'from pathlib import Path; from agri_vlm.schemas.config_schema import ModelConfigSchema, RLTrainConfigSchema, load_config; from agri_vlm.training.rl_trainer import validate_rl_sft_checkpoint_path; model=load_config(Path("$(RL_PHI4_MODEL_CONFIG)"), ModelConfigSchema); train=load_config(Path("$(RL_PHI4_FULL_CONFIG)"), RLTrainConfigSchema); validate_rl_sft_checkpoint_path(model, train); manifest=Path(train.manifest_path); assert manifest.exists(), "RL manifest missing: %s" % manifest'
	@echo "sbatch scripts/hpc/run_rl_grpo_phi4_full.slurm"

build-eval-manifest:
	PYTHONPATH=src $(PYTHON) scripts/data/build_eval_manifest.py \
		--config configs/data/eval_build.yaml \
		--download-mode $(DATA_DOWNLOAD_MODE) \
		--fraction $(DATA_FRACTION)

benchmark-status:
	PYTHONPATH=src:benchmarks/vlm_baselines $(PYTHON) scripts/benchmarks/benchmark_status.py \
		--phase both \
		--write-report

benchmark-phase-splits:
	PYTHONPATH=benchmarks/vlm_baselines $(PYTHON) benchmarks/vlm_baselines/build_phase_splits.py \
		--phase both \
		--write-report

benchmark-sft-splits:
	PYTHONPATH=benchmarks/vlm_baselines $(PYTHON) benchmarks/vlm_baselines/build_phase_splits.py \
		--phase sft \
		--write-report

benchmark-rl-splits:
	PYTHONPATH=benchmarks/vlm_baselines $(PYTHON) benchmarks/vlm_baselines/build_phase_splits.py \
		--phase rl \
		--write-report

benchmark-dry-run:
	$(MAKE) benchmark-sft-dry-run
	$(MAKE) benchmark-rl-dry-run

benchmark-sft-dry-run:
	PYTHONPATH=benchmarks/vlm_baselines $(PYTHON) benchmarks/vlm_baselines/run_baselines.py \
		--phase sft \
		--split val \
		--model-name "$(BENCHMARK_MODEL)" \
		--max-samples $(BENCHMARK_MAX_SAMPLES) \
		--dry-run

benchmark-rl-dry-run:
	PYTHONPATH=benchmarks/vlm_baselines $(PYTHON) benchmarks/vlm_baselines/run_baselines.py \
		--phase rl \
		--split val \
		--model-name "$(BENCHMARK_MODEL)" \
		--max-samples $(BENCHMARK_MAX_SAMPLES) \
		--dry-run

benchmark-sft-summary:
	PYTHONPATH=benchmarks/vlm_baselines $(PYTHON) benchmarks/vlm_baselines/evaluate_predictions.py \
		--refresh-summary-only \
		--output-dir benchmarks/vlm_baselines/results/metrics

benchmark-rl-summary: benchmark-sft-summary

benchmark-all-summary: benchmark-sft-summary

benchmark-sft-agvlm-checkpoint-dry-run:
	PYTHONPATH=benchmarks/vlm_baselines $(PYTHON) benchmarks/vlm_baselines/run_baselines.py \
		--phase sft \
		--split val \
		--model-key "$(BENCHMARK_MODEL_KEY)" \
		--max-samples $(BENCHMARK_MAX_SAMPLES) \
		--dry-run

benchmark-rl-agvlm-sft-checkpoint-dry-run:
	PYTHONPATH=benchmarks/vlm_baselines $(PYTHON) benchmarks/vlm_baselines/run_baselines.py \
		--phase rl \
		--split val \
		--model-key "$(BENCHMARK_MODEL_KEY)" \
		--max-samples $(BENCHMARK_MAX_SAMPLES) \
		--dry-run

benchmark-rl-agvlm-rl-checkpoint-dry-run:
	PYTHONPATH=benchmarks/vlm_baselines $(PYTHON) benchmarks/vlm_baselines/run_baselines.py \
		--phase rl \
		--split val \
		--model-key "$(BENCHMARK_RL_MODEL_KEY)" \
		--max-samples $(BENCHMARK_MAX_SAMPLES) \
		--dry-run

benchmark-rl-agvlm-checkpoint-dry-run: benchmark-rl-agvlm-rl-checkpoint-dry-run

export-training-artifacts:
	test -n "$(RUN_DIR)"
	PYTHONPATH=src $(PYTHON) scripts/artifacts/export_training_artifacts.py \
		--run-dir "$(RUN_DIR)"

export-benchmark-tables:
	test -n "$(BENCHMARK_RUNS)"
	PYTHONPATH=src $(PYTHON) scripts/artifacts/export_benchmark_tables.py $(BENCHMARK_RUNS)

data-report:
	PYTHONPATH=src $(PYTHON) scripts/data/dataset_report.py \
		--download-mode $(DATA_DOWNLOAD_MODE) \
		--fraction $(DATA_FRACTION)

sft:
	PYTHONPATH=src $(PYTHON) scripts/train/train_sft.py \
		--model-config configs/model/qwen_vlm_4b.yaml \
		--train-config configs/train/sft_lora.yaml

sft-dist:
	PYTHONPATH=src $(PYTHON) scripts/launch_torchrun.py \
		$(if $(NPROC_PER_NODE),--nproc-per-node $(NPROC_PER_NODE),) \
		--master-port $(MASTER_PORT) \
		scripts/train/train_sft.py -- \
		--model-config configs/model/qwen_vlm_4b.yaml \
		--train-config configs/train/sft_lora_b200_multigpu.yaml

rl:
	PYTHONPATH=src $(PYTHON) scripts/train/train_rl_grpo.py \
		--model-config configs/model/qwen_vlm_4b.yaml \
		--train-config configs/train/rl_grpo_lora.yaml

rl-dist:
	PYTHONPATH=src $(PYTHON) scripts/launch_torchrun.py \
		$(if $(NPROC_PER_NODE),--nproc-per-node $(NPROC_PER_NODE),) \
		--master-port $(MASTER_PORT) \
		scripts/train/train_rl_grpo.py -- \
		--model-config configs/model/qwen_vlm_4b.yaml \
		--train-config configs/train/rl_grpo_b200_multigpu.yaml

eval:
	PYTHONPATH=src $(PYTHON) scripts/eval/eval_local_holdout.py \
		--model-config configs/model/qwen_vlm_4b.yaml \
		--eval-config configs/eval/local_holdout.yaml

eval-local: eval

smoke:
	bash scripts/run_smoke_test.sh

test:
	PYTHONPATH=src $(PYTHON) -m pytest

test-rl:
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_reward_functions.py tests/test_rl_*.py -q
