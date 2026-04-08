#!/usr/bin/env bash
set -euo pipefail

module load conda
module load cuda/12.9.1

ENV_NAME="${ENV_NAME:-agri-vlm-v1}"
DOWNLOAD_MODE="${DOWNLOAD_MODE:-partial}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
export PYTHONPATH="${PYTHONPATH:-src}"
export AGRI_VLM_DATA_ROOT="${AGRI_VLM_DATA_ROOT:-$PWD/data}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$PWD/.tmp}"
mkdir -p "${AGRI_VLM_DATA_ROOT}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TMPDIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python scripts/verify_environment.py
python scripts/data/download_public_datasets.py --download-mode "${DOWNLOAD_MODE}" --fraction "${SAMPLE_FRACTION}"
python scripts/data/normalize_all.py --download-mode "${DOWNLOAD_MODE}" --fraction "${SAMPLE_FRACTION}"
python scripts/data/build_sft_manifest.py --download-mode "${DOWNLOAD_MODE}" --fraction "${SAMPLE_FRACTION}"
python scripts/data/build_rl_manifest.py --download-mode "${DOWNLOAD_MODE}" --fraction "${SAMPLE_FRACTION}"
python scripts/data/build_eval_manifest.py --download-mode "${DOWNLOAD_MODE}" --fraction "${SAMPLE_FRACTION}"
python scripts/data/dataset_report.py --download-mode "${DOWNLOAD_MODE}" --fraction "${SAMPLE_FRACTION}"
