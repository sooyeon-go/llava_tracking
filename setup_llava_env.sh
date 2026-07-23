#!/usr/bin/env bash
# Create a conda environment named "llava" for LLaVA-OneVision-2 tracking.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-llava}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
# Leave empty to auto-detect; examples: cu124, cu121, cpu
TORCH_CUDA="${TORCH_CUDA:-}"
# Set ALLOW_VENV=1 only if you intentionally want a local .venv fallback.
ALLOW_VENV="${ALLOW_VENV:-0}"

echo "[setup] ENV_NAME=${ENV_NAME}"
echo "[setup] SCRIPT_DIR=${SCRIPT_DIR}"

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  local candidate
  for candidate in \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda" \
    "${HOME}/mambaforge/bin/conda" \
    "${HOME}/miniforge3/bin/conda" \
    "/opt/conda/bin/conda" \
    "/usr/local/miniconda3/bin/conda"
  do
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

detect_torch_index() {
  if [[ -n "${TORCH_CUDA}" ]]; then
    case "${TORCH_CUDA}" in
      cpu) echo "https://download.pytorch.org/whl/cpu" ;;
      *) echo "https://download.pytorch.org/whl/${TORCH_CUDA}" ;;
    esac
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "https://download.pytorch.org/whl/cu124"
  else
    echo "https://download.pytorch.org/whl/cpu"
  fi
}

install_with_pip() {
  local python_bin="$1"
  local torch_index
  torch_index="$(detect_torch_index)"
  echo "[setup] Using Python: ${python_bin}"
  echo "[setup] Installing torch from: ${torch_index}"
  "${python_bin}" -m pip install --upgrade pip
  "${python_bin}" -m pip install \
    "torch>=2.4" torchvision \
    --index-url "${torch_index}"
  "${python_bin}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
}

verify_env() {
  local python_bin="$1"
  echo "[setup] Verifying imports..."
  "${python_bin}" - <<'PY'
import torch
import transformers
from PIL import Image
import numpy
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
print("imports_ok")
PY
}

CONDA_BIN=""
if CONDA_BIN="$(find_conda)"; then
  echo "[setup] Found conda: ${CONDA_BIN}"
  # shellcheck disable=SC1091
  source "$(dirname "${CONDA_BIN}")/../etc/profile.d/conda.sh"
  echo "[setup] Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup] Env already exists; updating packages."
  else
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
  conda activate "${ENV_NAME}"
  install_with_pip "$(command -v python)"
  verify_env "$(command -v python)"
  cat <<EOF

[setup] Done. Conda env ready: ${ENV_NAME}
Activate with:
  conda activate ${ENV_NAME}

Then run:
  cd ${SCRIPT_DIR}
  ./run_davis_tracking.sh
EOF
  exit 0
fi

if [[ "${ALLOW_VENV}" != "1" ]]; then
  cat <<EOF
[setup] ERROR: conda was not found on PATH.

This script installs into a conda env named "${ENV_NAME}" by default.
Load conda first, then re-run:

  source ~/miniconda3/etc/profile.d/conda.sh   # or your conda path
  ./setup_llava_env.sh

If you really want a local .venv instead:
  ALLOW_VENV=1 ./setup_llava_env.sh
EOF
  exit 1
fi

VENV_DIR="${SCRIPT_DIR}/.venv"
echo "[setup] ALLOW_VENV=1; creating venv at ${VENV_DIR}"
if [[ ! -d "${VENV_DIR}" ]]; then
  if command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
    "python${PYTHON_VERSION}" -m venv "${VENV_DIR}"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m venv "${VENV_DIR}"
  else
    python -m venv "${VENV_DIR}"
  fi
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
install_with_pip "$(command -v python)"
verify_env "$(command -v python)"
cat <<EOF

[setup] Done (venv fallback).
Activate with:
  source ${VENV_DIR}/bin/activate
EOF
