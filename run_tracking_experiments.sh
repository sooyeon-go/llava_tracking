#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${CONDA_DEFAULT_ENV:-}" && -z "${VIRTUAL_ENV:-}" && -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-/data/shared-vilab/pretrained_models/VLM_models/LLaVA-OneVision-2-8B-Instruct}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/data/shared-vilab/datasets/DAVIS/Annotations/Full-Resolution/bear}"
FRAMES_DIR="${FRAMES_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/davis_tracking_experiments/bear}"
DESCRIPTION="${DESCRIPTION:-the bear}"
SOURCE_FPS="${SOURCE_FPS:-24}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

args=(
  python
  "${SCRIPT_DIR}/llava_tracking_experiments.py"
  --model-path "${MODEL_PATH}"
  --annotation-dir "${ANNOTATION_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --description "${DESCRIPTION}"
  --source-fps "${SOURCE_FPS}"
  --device-map auto
  --dtype bfloat16
  --attn-implementation sdpa
)

if [[ -n "${FRAMES_DIR}" ]]; then
  args+=(--frames-dir "${FRAMES_DIR}")
fi

args+=("$@")

echo "Running: ${args[*]}"
"${args[@]}"
