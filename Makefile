PYTHON ?= python3
NPROC_PER_NODE ?=
MASTER_PORT ?= 29500
DATA_DOWNLOAD_MODE ?= partial
DATA_FRACTION ?= 0.1

.PHONY: help bootstrap verify verify-dist smoke prepare-slots download-data normalize-all data-partial data-full data-report build-sft-manifest build-rl-manifest build-eval-manifest benchmark-status export-training-artifacts export-benchmark-tables sft sft-dist rl rl-dist eval eval-local test

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
	@echo "  benchmark-status     Report benchmark readiness and missing paths"
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

build-eval-manifest:
	PYTHONPATH=src $(PYTHON) scripts/data/build_eval_manifest.py \
		--config configs/data/eval_build.yaml \
		--download-mode $(DATA_DOWNLOAD_MODE) \
		--fraction $(DATA_FRACTION)

benchmark-status:
	PYTHONPATH=src $(PYTHON) scripts/benchmarks/benchmark_status.py \
		--download-mode $(DATA_DOWNLOAD_MODE) \
		--fraction $(DATA_FRACTION)

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
