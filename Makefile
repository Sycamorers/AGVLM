PYTHON ?= python3
NPROC_PER_NODE ?=
MASTER_PORT ?= 29500

.PHONY: help bootstrap verify verify-dist smoke prepare-slots build-sft-manifest build-rl-manifest build-eval-manifest sft sft-dist rl rl-dist eval eval-local test

help:
	@echo "Targets:"
	@echo "  bootstrap            Create a local virtualenv with Python 3.11"
	@echo "  verify               Verify runtime and package prerequisites"
	@echo "  verify-dist          Verify a distributed torchrun environment"
	@echo "  smoke                Run the smoke test pipeline"
	@echo "  prepare-slots        Create manual dataset slots and smoke data"
	@echo "  build-sft-manifest   Build the SFT manifest"
	@echo "  build-rl-manifest    Build the RL manifest"
	@echo "  build-eval-manifest  Build evaluation manifests"
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
	PYTHONPATH=src $(PYTHON) scripts/data/prepare_manual_dataset_slots.py --with-smoke-data

build-sft-manifest:
	PYTHONPATH=src $(PYTHON) scripts/data/build_sft_manifest.py --config configs/data/sft_build.yaml

build-rl-manifest:
	PYTHONPATH=src $(PYTHON) scripts/data/build_rl_manifest.py --config configs/data/rl_build.yaml

build-eval-manifest:
	PYTHONPATH=src $(PYTHON) scripts/data/build_eval_manifest.py --config configs/data/eval_build.yaml

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
