#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" && -z "${VIRTUAL_ENV:-}" && -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-/data/shared-vilab/pretrained_models/VLM_models/LLaVA-OneVision-2-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/spair_correspondence_results}"
MAX_PAIRS="${MAX_PAIRS:-20}"
PAIR_SAMPLING="${PAIR_SAMPLING:-stratified}"
PCK_ALPHA="${PCK_ALPHA:-0.1}"
PROMPT_FORMAT="${PROMPT_FORMAT:-native-track}"
MIN_INPUT_PIXELS="${MIN_INPUT_PIXELS:-399360}"
SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-1}"
OVERWRITE="${OVERWRITE:-0}"
KEYPOINT_INDEX="${KEYPOINT_INDEX:-}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

args=(
  python
  "${SCRIPT_DIR}/llava_spair_correspondence.py"
  --model-path "${MODEL_PATH}"
  --dataset-root "${DATASET_ROOT}"
  --split test
  --layout-size large
  --max-pairs "${MAX_PAIRS}"
  --pair-sampling "${PAIR_SAMPLING}"
  --pck-alpha "${PCK_ALPHA}"
  --prompt-format "${PROMPT_FORMAT}"
  --min-input-pixels "${MIN_INPUT_PIXELS}"
  --output-dir "${OUTPUT_DIR}"
  --device-map auto
  --dtype bfloat16
  --attn-implementation sdpa
)

if [[ "${SAVE_VISUALIZATIONS}" == "1" ]]; then
  args+=(--save-visualizations)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ -n "${KEYPOINT_INDEX}" ]]; then
  args+=(--keypoint-index "${KEYPOINT_INDEX}")
fi

args+=("$@")

echo "Running: ${args[*]}"
"${args[@]}"
