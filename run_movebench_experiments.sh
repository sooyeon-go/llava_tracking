#!/usr/bin/env bash
# Extract MoveBench mp4s to frames, then run tracking experiment 1/2 + analysis.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VIDEO_ROOT="${VIDEO_ROOT:-/data/shared-vilab/datasets/MoveBench/en/video}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/movebench_tracking_results}"
RUN_EXPERIMENT1="${RUN_EXPERIMENT1:-1}"
RUN_EXPERIMENT2="${RUN_EXPERIMENT2:-1}"
SKIP_SHARED_HIGH_MOTION="${SKIP_SHARED_HIGH_MOTION:-1}"
MOVEMENT_THRESHOLD="${MOVEMENT_THRESHOLD:-1}"
MEANINGFUL_THRESHOLD="${MEANINGFUL_THRESHOLD:-10}"
EXTRACT_FRAMES="${EXTRACT_FRAMES:-1}"

videos=(
  "Pexels_3C_product_27.mp4"
  "Pexels_3C_product_10.mp4"
  "Pexels_3C_product_0.mp4"
)

# Referring expressions used by Track <description>
descriptions=(
  "the VR headset"
  "the VR headset"
  "the laptop"
)

sequence_names=(
  "Pexels_3C_product_27"
  "Pexels_3C_product_10"
  "Pexels_3C_product_0"
)

failed_runs=()

probe_fps() {
  local video_path="$1"
  local fps
  fps="$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=r_frame_rate \
    -of default=nokey=1:noprint_wrappers=1 \
    "${video_path}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${fps}" ]]; then
    echo "24"
    return
  fi
  python - "${fps}" <<'PY'
import sys
text = sys.argv[1].strip()
try:
    if "/" in text:
        num, den = text.split("/", 1)
        value = float(num) / float(den)
    else:
        value = float(text)
except Exception:
    value = 24.0
if value <= 0:
    value = 24.0
print(f"{value:.6f}".rstrip("0").rstrip("."))
PY
}

for index in "${!videos[@]}"; do
  video_name="${videos[$index]}"
  description="${descriptions[$index]}"
  sequence="${sequence_names[$index]}"
  video_path="${VIDEO_ROOT}/${video_name}"
  frames_dir="${VIDEO_ROOT}/${sequence}_frames"
  sequence_output="${RESULTS_ROOT}/${sequence}"

  echo
  echo "============================================================"
  echo "Sequence: ${sequence}"
  echo "Prompt  : Track ${description}"
  echo "Video   : ${video_path}"
  echo "Frames  : ${frames_dir}"
  echo "============================================================"

  if [[ ! -f "${video_path}" ]]; then
    echo "ERROR: video does not exist: ${video_path}" >&2
    failed_runs+=("${sequence}:missing-video")
    continue
  fi

  if [[ "${EXTRACT_FRAMES}" == "1" ]]; then
    if ! bash "${SCRIPT_DIR}/extract_webm_frames.sh" "${video_path}"; then
      echo "ERROR: frame extraction failed for ${sequence}" >&2
      failed_runs+=("${sequence}:extract-frames")
      continue
    fi
  fi

  if [[ ! -d "${frames_dir}" ]] || [[ -z "$(find "${frames_dir}" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) | head -n 1)" ]]; then
    echo "ERROR: no extracted frames in ${frames_dir}" >&2
    failed_runs+=("${sequence}:missing-frames")
    continue
  fi

  source_fps="$(probe_fps "${video_path}")"
  echo "Detected FPS: ${source_fps}"

  # No mask annotations for MoveBench; experiments only need FRAMES_DIR.
  annotation_dir="${frames_dir}"

  if [[ "${RUN_EXPERIMENT1}" == "1" ]]; then
    echo "[${sequence}] Starting experiment 1"
    if ! FRAMES_DIR="${frames_dir}" \
      ANNOTATION_DIR="${annotation_dir}" \
      DESCRIPTION="${description}" \
      OUTPUT_DIR="${sequence_output}/experiment1" \
      SOURCE_FPS="${source_fps}" \
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
      SOURCE_FPS="${source_fps}" \
      SKIP_HIGH_MOTION="${SKIP_SHARED_HIGH_MOTION}" \
      bash "${SCRIPT_DIR}/run_tracking_experiments2.sh"; then
      echo "ERROR: experiment 2 failed for ${sequence}" >&2
      failed_runs+=("${sequence}:experiment2")
    fi
  fi
done

echo
echo "Frames stay next to the mp4s under: ${VIDEO_ROOT}"
echo "Experiment outputs: ${RESULTS_ROOT}"
echo "Analyzing predicted point movement across all completed runs"
if ! python "${SCRIPT_DIR}/analyze_tracking_movements.py" \
  --results-root "${RESULTS_ROOT}" \
  --movement-threshold "${MOVEMENT_THRESHOLD}" \
  --meaningful-threshold "${MEANINGFUL_THRESHOLD}"; then
  echo "ERROR: movement analysis failed" >&2
  failed_runs+=("movement-analysis")
fi

if (( ${#failed_runs[@]} > 0 )); then
  echo "Failed or skipped runs:" >&2
  printf '  - %s\n' "${failed_runs[@]}" >&2
  exit 1
fi

echo "All MoveBench experiments completed."
