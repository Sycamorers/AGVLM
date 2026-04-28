#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Required interpreter not found: ${PYTHON_BIN}" >&2
  echo "Install Python 3.11 and rerun, or set PYTHON_BIN=/path/to/python3.11" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m ensurepip --upgrade
python -m pip install --upgrade pip wheel "setuptools<81"

echo "Installing PyTorch ${TORCH_VERSION} wheels from ${TORCH_INDEX_URL}"
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

python -m pip install -e ".[dev,qwen-utils,deepspeed]"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "Installing flash-attn from pip. This path should be validated on the target CUDA 12.9.1 / B200 image."
  python -m pip install "flash-attn>=2.8.0.post2"
fi

echo "Environment bootstrapped in ${VENV_DIR}"
echo "This bootstrap flow assumes an NVIDIA driver stack compatible with CUDA 12.9.1."
