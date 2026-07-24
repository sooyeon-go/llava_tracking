#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" && -z "${VIRTUAL_ENV:-}" && -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

# Comma-separated GPU ids, e.g. GPU_IDS=0,1,2,3
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
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

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_SHARDS="${#GPU_ARRAY[@]}"
if (( NUM_SHARDS < 1 )); then
  echo "ERROR: GPU_IDS is empty" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
if [[ "${OVERWRITE}" == "1" ]]; then
  rm -f "${OUTPUT_DIR}/test_predictions.jsonl" "${OUTPUT_DIR}/test_summary.json"
  rm -rf "${OUTPUT_DIR}/shards"
fi

extra_args=()
if [[ "${SAVE_VISUALIZATIONS}" == "1" ]]; then
  extra_args+=(--save-visualizations)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  extra_args+=(--overwrite)
fi
if [[ -n "${KEYPOINT_INDEX}" ]]; then
  extra_args+=(--keypoint-index "${KEYPOINT_INDEX}")
fi
extra_args+=("$@")

pids=()
shard_id=0
for gpu_id in "${GPU_ARRAY[@]}"; do
  log_file="${OUTPUT_DIR}/shard$(printf '%02d' "${shard_id}").gpu${gpu_id}.log"
  echo "Launching shard ${shard_id}/${NUM_SHARDS} on GPU ${gpu_id}"
  echo "  log: ${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    python "${SCRIPT_DIR}/llava_spair_correspondence.py" \
      --model-path "${MODEL_PATH}" \
      --dataset-root "${DATASET_ROOT}" \
      --split test \
      --layout-size large \
      --max-pairs "${MAX_PAIRS}" \
      --pair-sampling "${PAIR_SAMPLING}" \
      --pck-alpha "${PCK_ALPHA}" \
      --prompt-format "${PROMPT_FORMAT}" \
      --min-input-pixels "${MIN_INPUT_PIXELS}" \
      --output-dir "${OUTPUT_DIR}" \
      --device-map "cuda:0" \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --num-shards "${NUM_SHARDS}" \
      --shard-id "${shard_id}" \
      "${extra_args[@]}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
  shard_id=$((shard_id + 1))
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "ERROR: one or more GPU shards failed. Check ${OUTPUT_DIR}/shard*.log" >&2
  exit 1
fi

merged="${OUTPUT_DIR}/test_predictions.jsonl"
if (( NUM_SHARDS > 1 )); then
  : > "${merged}"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    shard_file="$(printf '%s/shards/shard%02d/test_predictions.jsonl' "${OUTPUT_DIR}" "${i}")"
    if [[ -f "${shard_file}" ]]; then
      cat "${shard_file}" >> "${merged}"
    fi
  done
fi

python "${SCRIPT_DIR}/llava_spair_correspondence.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split test \
  --summary-only

echo "Merged predictions: ${merged}"
echo "Summary: ${OUTPUT_DIR}/test_summary.json"
