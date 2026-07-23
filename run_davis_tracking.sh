#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Prefer an already-activated env; otherwise use local .venv if present.
if [[ -z "${CONDA_DEFAULT_ENV:-}" && -z "${VIRTUAL_ENV:-}" && -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-/data/shared-vilab/pretrained_models/VLM_models/LLaVA-OneVision-2-8B-Instruct}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/data/shared-vilab/datasets/DAVIS/Annotations/Full-Resolution/bear}"
FRAMES_DIR="${FRAMES_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/davis_tracking_results/bear}"
MAX_FRAMES="${MAX_FRAMES:-16}"
FRAME_SAMPLING="${FRAME_SAMPLING:-uniform}"
DESCRIPTION="${DESCRIPTION:-the bear}"
TRACKING_MODE="${TRACKING_MODE:-text-grounding}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

args=(
  python
  "${SCRIPT_DIR}/llava_davis_tracking.py"
  --model-path "${MODEL_PATH}"
  --annotation-dir "${ANNOTATION_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --description "${DESCRIPTION}"
  --tracking-mode "${TRACKING_MODE}"
  --max-frames "${MAX_FRAMES}"
  --frame-sampling "${FRAME_SAMPLING}"
  --device-map auto
  --dtype bfloat16
  --attn-implementation sdpa
  --save-visualizations
  --save-overlay-gif
)

if [[ -n "${FRAMES_DIR}" ]]; then
  args+=(--frames-dir "${FRAMES_DIR}")
fi

args+=("$@")

echo "Running: ${args[*]}"
"${args[@]}"
