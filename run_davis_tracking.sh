#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-/data/shared-vilab/pretrained_models/VLM_models/LLaVA-OneVision-2-8B-Instruct}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/data/shared-vilab/datasets/DAVIS/Annotations/Full-Resolution/bear}"
FRAMES_DIR="${FRAMES_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/davis_tracking_results/bear}"
MAX_FRAMES="${MAX_FRAMES:-16}"
TARGET_INPUT_PIXELS="${TARGET_INPUT_PIXELS:-399360}"  # 832 * 480

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

args=(
  python
  "${SCRIPT_DIR}/llava_davis_tracking.py"
  --model-path "${MODEL_PATH}"
  --annotation-dir "${ANNOTATION_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --max-frames "${MAX_FRAMES}"
  --target-input-pixels "${TARGET_INPUT_PIXELS}"
  --device-map auto
  --dtype bfloat16
  --attn-implementation sdpa
  --save-visualizations
)

if [[ -n "${FRAMES_DIR}" ]]; then
  args+=(--frames-dir "${FRAMES_DIR}")
fi

args+=("$@")

echo "Running: ${args[*]}"
"${args[@]}"
