#!/usr/bin/env bash
# Extract every frame from a .webm/.mp4 into <video_stem>_frames/ next to the file.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/video.webm [/path/to/other.webm ...]" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found on PATH" >&2
  exit 1
fi

for video_path in "$@"; do
  if [[ ! -f "${video_path}" ]]; then
    echo "ERROR: video does not exist: ${video_path}" >&2
    exit 1
  fi

  video_dir="$(cd -- "$(dirname -- "${video_path}")" && pwd)"
  video_base="$(basename -- "${video_path}")"
  stem="${video_base%.*}"
  frames_dir="${video_dir}/${stem}_frames"

  mkdir -p "${frames_dir}"
  # Clear previous extraction so numbering stays contiguous.
  find "${frames_dir}" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) -delete

  echo "Extracting frames:"
  echo "  video : ${video_dir}/${video_base}"
  echo "  frames: ${frames_dir}"

  # Keep original cadence; write zero-padded PNGs for stable sorting.
  ffmpeg -hide_banner -loglevel error -y \
    -i "${video_dir}/${video_base}" \
    -vsync 0 \
    "${frames_dir}/%05d.png"

  frame_count="$(find "${frames_dir}" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
  echo "  wrote ${frame_count} frames"
done
