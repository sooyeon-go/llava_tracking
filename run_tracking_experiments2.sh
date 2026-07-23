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
DESCRIPTION="${DESCRIPTION:-the bear}"
MOTION_ANNOTATION_DIR="${MOTION_ANNOTATION_DIR:-/data/shared-vilab/datasets/DAVIS/Annotations/Full-Resolution/drift-chicane}"
MOTION_FRAMES_DIR="${MOTION_FRAMES_DIR:-}"
MOTION_DESCRIPTION="${MOTION_DESCRIPTION:-a sport car}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/davis_tracking_experiments2}"
SOURCE_FPS="${SOURCE_FPS:-24}"
MOTION_SOURCE_FPS="${MOTION_SOURCE_FPS:-24}"
OVERLAY_FPS="${OVERLAY_FPS:-4}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

args=(
  python
  "${SCRIPT_DIR}/llava_tracking_experiments2.py"
  --model-path "${MODEL_PATH}"
  --annotation-dir "${ANNOTATION_DIR}"
  --description "${DESCRIPTION}"
  --motion-annotation-dir "${MOTION_ANNOTATION_DIR}"
  --motion-description "${MOTION_DESCRIPTION}"
  --output-dir "${OUTPUT_DIR}"
  --source-fps "${SOURCE_FPS}"
  --motion-source-fps "${MOTION_SOURCE_FPS}"
  --overlay-fps "${OVERLAY_FPS}"
  --device-map auto
  --dtype bfloat16
  --attn-implementation sdpa
)

if [[ -n "${FRAMES_DIR}" ]]; then
  args+=(--frames-dir "${FRAMES_DIR}")
fi
if [[ -n "${MOTION_FRAMES_DIR}" ]]; then
  args+=(--motion-frames-dir "${MOTION_FRAMES_DIR}")
fi
if [[ "${SKIP_HIGH_MOTION:-0}" == "1" ]]; then
  args+=(--skip-high-motion)
fi

args+=("$@")

echo "Running: ${args[*]}"
"${args[@]}"
