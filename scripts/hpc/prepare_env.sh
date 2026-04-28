#!/usr/bin/env bash
set -euo pipefail

module load conda
module load cuda/12.9.1

ENV_NAME="${ENV_NAME:-agri-vlm-v1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
python -m pip install -e ".[dev,qwen-utils,deepspeed]"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  python -m pip install "flash-attn>=2.8.0.post2"
fi

export AGRI_VLM_DATA_ROOT="${AGRI_VLM_DATA_ROOT:-$PWD/data}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$PWD/.tmp}"
mkdir -p "${AGRI_VLM_DATA_ROOT}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TMPDIR}"

echo "Prepared conda environment ${ENV_NAME}"
echo "AGRI_VLM_DATA_ROOT=${AGRI_VLM_DATA_ROOT}"
