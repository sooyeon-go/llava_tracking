#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DAVIS_ROOT="${DAVIS_ROOT:-/data/shared-vilab/datasets/DAVIS}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/davis_tracking_multi_sequence}"
RUN_EXPERIMENT1="${RUN_EXPERIMENT1:-1}"
RUN_EXPERIMENT2="${RUN_EXPERIMENT2:-1}"
SKIP_SHARED_HIGH_MOTION="${SKIP_SHARED_HIGH_MOTION:-1}"

sequences=(
  "boat"
  "car-shadow"
  "dog-agility"
  "motorbike"
  "tennis"
)

descriptions=(
  "the boat"
  "the car"
  "the dog"
  "the motorbike"
  "the person playing tennis"
)

failed_runs=()

for index in "${!sequences[@]}"; do
  sequence="${sequences[$index]}"
  description="${descriptions[$index]}"
  frames_dir="${DAVIS_ROOT}/JPEGImages/Full-Resolution/${sequence}"
  annotation_dir="${DAVIS_ROOT}/Annotations/Full-Resolution/${sequence}"
  sequence_output="${RESULTS_ROOT}/${sequence}"

  echo
  echo "============================================================"
  echo "Sequence: ${sequence}"
  echo "Prompt: Track ${description}"
  echo "Frames: ${frames_dir}"
  echo "============================================================"

  if [[ ! -d "${frames_dir}" ]]; then
    echo "ERROR: frames directory does not exist: ${frames_dir}" >&2
    failed_runs+=("${sequence}:missing-frames")
    continue
  fi

  if [[ "${RUN_EXPERIMENT1}" == "1" ]]; then
    echo "[${sequence}] Starting experiment 1"
    if ! FRAMES_DIR="${frames_dir}" \
      ANNOTATION_DIR="${annotation_dir}" \
      DESCRIPTION="${description}" \
      OUTPUT_DIR="${sequence_output}/experiment1" \
      bash "${SCRIPT_DIR}/run_tracking_experiments.sh"; then
      echo "ERROR: experiment 1 failed for ${sequence}" >&2
      failed_runs+=("${sequence}:experiment1")
    fi
  fi

  if [[ "${RUN_EXPERIMENT2}" == "1" ]]; then
    echo "[${sequence}] Starting experiment 2"
    if ! FRAMES_DIR="${frames_dir}" \
      ANNOTATION_DIR="${annotation_dir}" \
      DESCRIPTION="${description}" \
      OUTPUT_DIR="${sequence_output}/experiment2" \
      SKIP_HIGH_MOTION="${SKIP_SHARED_HIGH_MOTION}" \
      bash "${SCRIPT_DIR}/run_tracking_experiments2.sh"; then
      echo "ERROR: experiment 2 failed for ${sequence}" >&2
      failed_runs+=("${sequence}:experiment2")
    fi
  fi
done

echo
echo "Results root: ${RESULTS_ROOT}"
if (( ${#failed_runs[@]} > 0 )); then
  echo "Failed or skipped runs:" >&2
  printf '  - %s\n' "${failed_runs[@]}" >&2
  exit 1
fi

echo "All requested sequence experiments completed."
