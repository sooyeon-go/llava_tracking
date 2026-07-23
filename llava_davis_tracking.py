#!/usr/bin/env python3
"""Track DAVIS objects from a folder of frames with LLaVA-OneVision-2.

The script accepts a DAVIS annotation folder, finds the matching RGB frame
folder, extracts one starting point per object from the first mask, and asks
LLaVA-OneVision-2 to emit trajectories in its native ``<tracks>`` grammar.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

LOGGER = logging.getLogger("llava_davis_tracking")
COORDINATE_SCALE = 1000.0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = (
    "red",
    "lime",
    "cyan",
    "yellow",
    "magenta",
    "orange",
    "deepskyblue",
    "white",
)


def point_argument(value: str) -> tuple[float, float]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return float(x_text), float(y_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("point must have the form X,Y") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/data/shared-vilab/pretrained_models/VLM_models/"
            "LLaVA-OneVision-2-8B-Instruct"
        ),
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path(
            "/data/shared-vilab/datasets/DAVIS/"
            "Annotations/Full-Resolution/bear"
        ),
        help="DAVIS mask folder. Used to infer points and the RGB frame folder.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help=(
            "RGB frame folder. If omitted, replace Annotations with JPEGImages "
            "in --annotation-dir."
        ),
    )
    parser.add_argument(
        "--point",
        type=point_argument,
        action="append",
        help="Manual first-frame point in original pixels, X,Y. Repeat as needed.",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        action="append",
        help="Only initialize these DAVIS mask values. Repeat as needed.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=8,
        help="Maximum number of mask objects to initialize.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Text description of what to track; defaults to the folder name.",
    )
    parser.add_argument(
        "--tracking-mode",
        choices=("text-grounding", "point-prompt"),
        default="point-prompt",
        help=(
            "point-prompt initializes tracking from a mask/manual point; "
            "text-grounding finds the described object from text alone."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Uniformly sample at most this many frames; 0 keeps every frame.",
    )
    parser.add_argument("--marker-radius", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("davis_tracking_results/bear"),
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save RGB frames overlaid with predicted points and trajectories.",
    )
    parser.add_argument(
        "--save-overlay-gif",
        action="store_true",
        help="Also save a downscaled animated GIF of the overlays.",
    )
    parser.add_argument("--overlay-fps", type=float, default=4.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare frames and prompt without loading the model.",
    )
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if args.max_objects <= 0:
        parser.error("--max-objects must be > 0")
    if args.overlay_fps <= 0:
        parser.error("--overlay-fps must be > 0")
    if args.tracking_mode == "text-grounding" and args.point:
        parser.error("--point can only be used with --tracking-mode point-prompt")
    return args


def natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {directory}")
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    paths.sort(key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No image files found in: {directory}")
    return paths


def infer_frames_dir(annotation_dir: Path) -> Path:
    parts = list(annotation_dir.parts)
    try:
        annotation_index = parts.index("Annotations")
    except ValueError as error:
        raise ValueError(
            "--frames-dir is required when --annotation-dir does not contain "
            "an 'Annotations' path component"
        ) from error
    parts[annotation_index] = "JPEGImages"
    return Path(*parts)


def uniformly_sample(paths: list[Path], max_frames: int) -> tuple[list[Path], list[int]]:
    if max_frames == 0 or len(paths) <= max_frames:
        return paths, list(range(len(paths)))
    indices = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
    unique_indices = list(dict.fromkeys(indices.tolist()))
    return [paths[index] for index in unique_indices], unique_indices


def matching_mask(annotation_dir: Path, frame_path: Path) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = annotation_dir / f"{frame_path.stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No annotation matching frame {frame_path.name} in {annotation_dir}"
    )


def point_inside_mask_near_centroid(
    mask: np.ndarray,
    mask_value: int,
) -> tuple[float, float]:
    y_coords, x_coords = np.nonzero(mask == mask_value)
    if len(x_coords) == 0:
        raise ValueError(f"Mask value {mask_value} has no pixels")
    center_x = float(x_coords.mean())
    center_y = float(y_coords.mean())
    nearest = np.argmin(
        np.square(x_coords - center_x) + np.square(y_coords - center_y)
    )
    return float(x_coords[nearest]), float(y_coords[nearest])


def points_from_mask(
    mask_path: Path,
    requested_values: list[int] | None,
    max_objects: int,
) -> tuple[list[tuple[float, float]], list[int]]:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    available_values = [
        int(value) for value in np.unique(mask) if int(value) != 0
    ]
    if requested_values:
        missing = sorted(set(requested_values) - set(available_values))
        if missing:
            raise ValueError(
                f"Mask values {missing} are absent from {mask_path}; "
                f"available={available_values}"
            )
        selected_values = requested_values
    else:
        selected_values = available_values[:max_objects]
    if not selected_values:
        raise ValueError(f"No foreground objects found in mask: {mask_path}")
    points = [
        point_inside_mask_near_centroid(mask, value)
        for value in selected_values
    ]
    return points, selected_values


def prepare_frames(
    frame_paths: list[Path],
) -> tuple[list[Image.Image], tuple[int, int]]:
    """Load frames as-is; LLaVA's video processor chooses the model resolution."""
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    original_size = frames[0].size
    inconsistent = [
        (path.name, frame.size)
        for path, frame in zip(frame_paths, frames)
        if frame.size != original_size
    ]
    if inconsistent:
        raise ValueError(
            "All frames must have the same resolution; mismatches: "
            f"{inconsistent[:5]}"
        )
    return frames, original_size


def normalized_point(
    point: Iterable[float],
    image_size: tuple[int, int],
) -> tuple[int, int]:
    x, y = (float(value) for value in point)
    width, height = image_size
    return (
        round(x / max(width - 1, 1) * COORDINATE_SCALE),
        round(y / max(height - 1, 1) * COORDINATE_SCALE),
    )


def normalized_to_pixel(
    point: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    x, y = point
    if not (0 <= x <= COORDINATE_SCALE and 0 <= y <= COORDINATE_SCALE):
        return None
    width, height = image_size
    return (
        x / COORDINATE_SCALE * (width - 1),
        y / COORDINATE_SCALE * (height - 1),
    )


def draw_markers(
    image: Image.Image,
    points: list[tuple[float, float]],
    radius: int,
) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for point_id, (x, y) in enumerate(points):
        color = COLORS[point_id % len(COLORS)]
        width = max(3, radius // 3)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=width,
        )
        draw.line((x - 2 * radius, y, x + 2 * radius, y), fill=color, width=width)
        draw.line((x, y - 2 * radius, x, y + 2 * radius), fill=color, width=width)
    return marked


def build_point_prompt(
    description: str,
    points: list[tuple[float, float]],
    image_size: tuple[int, int],
    num_frames: int,
) -> str:
    normalized = [normalized_point(point, image_size) for point in points]
    source_fields = " ".join(
        f"{point_id} {x} {y}"
        for point_id, (x, y) in enumerate(normalized)
    )
    timestamps = ", ".join(f"{index:.1f}" for index in range(num_frames))
    source_track = (
        f'<tracks coords="0.0 {source_fields}">'
        f"{description} source points</tracks>"
    )
    return (
        f"Track the marked points corresponding to '{description}' through the "
        "entire source video. The points are marked only on the first frame. "
        "Point coordinates are integers from 0 to 1000 relative to each full "
        "frame. Keep the same 0-based point IDs. The source points are:\n"
        f"{source_track}\n"
        "For EVERY later timestamp, output the updated absolute position of the "
        "same physical point after the object moves. Do not copy the first-frame "
        "coordinates into later frames unless the object truly did not move. "
        "Return each point's absolute position at every supplied frame in "
        "OneVision2's native frame-major grammar. Frame groups must be separated "
        "by semicolons, and each group is: timestamp point_id x y [point_id x y "
        "...]. Use exactly these timestamps:\n"
        f"{timestamps}\n"
        "Answer with exactly one <tracks> element and no explanation."
    )


def build_grounding_prompt(description: str, num_frames: int) -> str:
    """Prompt for text-only object grounding followed by point tracking."""
    timestamps = ", ".join(f"{index:.1f}" for index in range(num_frames))
    return (
        f"Ground the complete object matching this description in the source "
        f"video: '{description}'. First localize the described object in frame "
        "0.0 using only the text and visual content; there is no input point or "
        "marker. Choose one representative point near the center and inside the "
        "object, assign it point id 0, and track that same physical object point "
        "through every supplied frame. If multiple distinct objects match the "
        "description, assign consecutive 0-based point IDs and track one center "
        "point per object. Coordinates must be absolute integer positions from "
        "0 to 1000 relative to each full frame. Return updated coordinates at "
        "EVERY timestamp; do not copy a coordinate to later frames unless the "
        "object truly did not move. Use OneVision2's native frame-major grammar "
        "with semicolon-separated groups: timestamp point_id x y [point_id x y "
        "...]. Use exactly these timestamps:\n"
        f"{timestamps}\n"
        "Answer with exactly one <tracks> element and no explanation."
    )


def parse_tracks(text: str) -> dict[float, dict[int, tuple[float, float]]]:
    matches = re.findall(
        r"<tracks\b[^>]*\bcoords\s*=\s*[\"']([^\"']+)[\"'][^>]*>",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        return {}
    tracks: dict[float, dict[int, tuple[float, float]]] = {}
    for frame_group in matches[-1].split(";"):
        fields = frame_group.strip().split()
        if len(fields) < 4:
            continue
        try:
            timestamp = float(fields[0])
        except ValueError:
            continue
        frame_points: dict[int, tuple[float, float]] = {}
        point_fields = fields[1:]
        for index in range(0, len(point_fields) - 2, 3):
            try:
                point_id = int(float(point_fields[index]))
                x = float(point_fields[index + 1])
                y = float(point_fields[index + 2])
            except ValueError:
                continue
            if 0 <= x <= COORDINATE_SCALE and 0 <= y <= COORDINATE_SCALE:
                frame_points[point_id] = (x, y)
        if frame_points:
            tracks[timestamp] = frame_points
    return tracks


class LlavaVideoTracker:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        dtype = getattr(torch, args.dtype)
        LOGGER.info("Loading processor from %s", args.model_path)
        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        LOGGER.info("Loading model from %s", args.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        ).eval()
        self.max_new_tokens = args.max_new_tokens

    def _input_device(self) -> Any:
        return next(self.model.parameters()).device

    def predict(self, frames: list[Image.Image], prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[chat_text],
            videos=frames,
            padding=True,
            return_tensors="pt",
        )
        device = self._input_device()
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        tokenizer = self.processor.tokenizer
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        return self.processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def mask_hit(
    mask_path: Path,
    pixel_point: tuple[float, float],
    mask_value: int,
) -> bool:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    x = int(round(pixel_point[0]))
    y = int(round(pixel_point[1]))
    x = min(max(x, 0), mask.shape[1] - 1)
    y = min(max(y, 0), mask.shape[0] - 1)
    return int(mask[y, x]) == mask_value


def ground_truth_centroid(
    annotation_dir: Path,
    frame_path: Path,
    mask_value: int,
) -> tuple[float, float] | None:
    try:
        mask_path = matching_mask(annotation_dir, frame_path)
    except FileNotFoundError:
        return None
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    y_coords, x_coords = np.nonzero(mask == mask_value)
    if len(x_coords) == 0:
        return None
    return float(x_coords.mean()), float(y_coords.mean())


def trajectory_span(results: list[dict[str, Any]]) -> dict[int, float]:
    spans: dict[int, float] = {}
    histories: dict[int, list[tuple[float, float]]] = {}
    for frame in results:
        for point in frame["points"]:
            if point["pixel_xy"] is None:
                continue
            point_id = int(point["point_id"])
            histories.setdefault(point_id, []).append(
                (float(point["pixel_xy"][0]), float(point["pixel_xy"][1]))
            )
    for point_id, coords in histories.items():
        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]
        spans[point_id] = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return spans


def build_results(
    tracks: dict[float, dict[int, tuple[float, float]]],
    frame_paths: list[Path],
    original_indices: list[int],
    original_size: tuple[int, int],
    annotation_dir: Path | None,
    mask_values: list[int | None],
) -> list[dict[str, Any]]:
    results = []
    for sampled_index, (frame_path, original_index) in enumerate(
        zip(frame_paths, original_indices)
    ):
        frame_points = tracks.get(float(sampled_index), {})
        points = []
        for point_id, mask_value in enumerate(mask_values):
            normalized = frame_points.get(point_id)
            pixel = (
                normalized_to_pixel(normalized, original_size)
                if normalized is not None
                else None
            )
            hit = None
            gt_centroid = None
            if annotation_dir is not None and mask_value is not None:
                gt_centroid = ground_truth_centroid(
                    annotation_dir, frame_path, mask_value
                )
                if pixel is not None:
                    try:
                        hit = mask_hit(
                            matching_mask(annotation_dir, frame_path),
                            pixel,
                            mask_value,
                        )
                    except FileNotFoundError:
                        hit = None
            points.append(
                {
                    "point_id": point_id,
                    "mask_value": mask_value,
                    "normalized_xy": list(normalized) if normalized else None,
                    "pixel_xy": list(pixel) if pixel else None,
                    "gt_centroid_xy": list(gt_centroid) if gt_centroid else None,
                    "inside_ground_truth_mask": hit,
                }
            )
        results.append(
            {
                "sampled_index": sampled_index,
                "model_timestamp": float(sampled_index),
                "original_frame_index": original_index,
                "frame": str(frame_path),
                "points": points,
            }
        )
    return results


def save_visualizations(
    frame_paths: list[Path],
    results: list[dict[str, Any]],
    output_dir: Path,
    radius: int,
    description: str,
    gif_path: Path | None,
    overlay_fps: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_frames: list[Image.Image] = []
    for frame_path, frame_result in zip(frame_paths, results):
        image = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        title = (
            f"{description} | sampled {frame_result['sampled_index']} "
            f"(source #{frame_result['original_frame_index']})"
        )
        title_box = draw.textbbox((0, 0), title)
        draw.rectangle(
            (0, 0, title_box[2] + 12, title_box[3] + 10),
            fill=(0, 0, 0),
        )
        draw.text((6, 5), title, fill="white")
        for point in frame_result["points"]:
            if point["pixel_xy"] is None:
                continue
            point_id = int(point["point_id"])
            color = COLORS[point_id % len(COLORS)]
            x, y = point["pixel_xy"]
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=color,
                width=4,
            )
            draw.line((x - radius, y, x + radius, y), fill=color, width=4)
            draw.line((x, y - radius, x, y + radius), fill=color, width=4)
            draw.text((x + radius + 3, y - radius), f"pred{point_id}", fill=color)
        image.save(output_dir / f"{frame_path.stem}.png")
        if gif_path is not None:
            preview = image.copy()
            if preview.width > 832:
                preview.thumbnail((832, 832), Image.Resampling.LANCZOS)
            gif_frames.append(preview)

    if gif_path is not None and gif_frames:
        duration_ms = round(1000 / overlay_fps)
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    annotation_dir = args.annotation_dir.resolve() if args.annotation_dir else None
    frames_dir = (
        args.frames_dir.resolve()
        if args.frames_dir
        else infer_frames_dir(annotation_dir)
    )
    all_frame_paths = image_files(frames_dir)
    frame_paths, original_indices = uniformly_sample(
        all_frame_paths,
        args.max_frames,
    )
    frames, original_size = prepare_frames(frame_paths)
    description = args.description or frames_dir.name
    LOGGER.info(
        "Frames: %d/%d, frame_size=%s (processor chooses model resolution)",
        len(frame_paths),
        len(all_frame_paths),
        original_size,
    )

    first_mask = (
        matching_mask(annotation_dir, frame_paths[0])
        if annotation_dir is not None
        else None
    )
    if args.tracking_mode == "text-grounding":
        source_points: list[tuple[float, float]] = []
        if first_mask is not None:
            _, selected_mask_values = points_from_mask(
                first_mask,
                args.object_id,
                args.max_objects,
            )
            mask_values: list[int | None] = list(selected_mask_values)
            LOGGER.info(
                "Text-grounding mode: annotations are evaluation-only; "
                "mask values=%s are not passed to the model",
                mask_values,
            )
        else:
            mask_values = [None]
        prompt = build_grounding_prompt(description, len(frames))
    else:
        if args.point:
            source_points = list(args.point)
            mask_values = [None] * len(source_points)
        else:
            if first_mask is None:
                raise ValueError(
                    "point-prompt mode requires --annotation-dir or --point"
                )
            source_points, selected_mask_values = points_from_mask(
                first_mask,
                args.object_id,
                args.max_objects,
            )
            mask_values = list(selected_mask_values)
            LOGGER.info(
                "Initialized model input points from %s: %s",
                first_mask,
                list(zip(mask_values, source_points)),
            )
        frames[0] = draw_markers(frames[0], source_points, args.marker_radius)
        prompt = build_point_prompt(
            description,
            source_points,
            original_size,
            len(frames),
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt + "\n")
    frames[0].save(output_dir / "input_first_frame.png")
    if args.dry_run:
        LOGGER.info("Dry run complete: %s", output_dir)
        return

    tracker = LlavaVideoTracker(args)
    raw_output = tracker.predict(frames, prompt)
    (output_dir / "raw_output.txt").write_text(raw_output + "\n")
    tracks = parse_tracks(raw_output)
    if not tracks:
        LOGGER.warning("No native tracks could be parsed from model output")
    results = build_results(
        tracks,
        frame_paths,
        original_indices,
        original_size,
        annotation_dir,
        mask_values,
    )
    valid_hits = [
        point["inside_ground_truth_mask"]
        for frame in results
        for point in frame["points"]
        if point["inside_ground_truth_mask"] is not None
    ]
    spans = trajectory_span(results)
    expected_point_count = len(mask_values)
    for point_id, span in spans.items():
        if span < 1.0:
            LOGGER.warning(
                "Predicted trajectory for point_id=%d is nearly static "
                "(span=%.2f px). Visualization is showing a fixed point because "
                "the model copied the same coordinates across frames. Check "
                "raw_output.txt.",
                point_id,
                span,
            )
    payload = {
        "sequence": frames_dir.name,
        "description": description,
        "tracking_mode": args.tracking_mode,
        "frames_dir": str(frames_dir),
        "annotation_dir": str(annotation_dir) if annotation_dir else None,
        "original_size": list(original_size),
        "num_source_frames": len(all_frame_paths),
        "num_sampled_frames": len(frame_paths),
        "source_points": [list(point) for point in source_points],
        "mask_values": mask_values,
        "prediction_span_px": {str(key): value for key, value in spans.items()},
        "coverage": (
            sum(
                point["pixel_xy"] is not None
                for frame in results
                for point in frame["points"]
            )
            / (len(results) * expected_point_count)
        ),
        "mask_hit_rate": (
            sum(valid_hits) / len(valid_hits) if valid_hits else None
        ),
        "frames": results,
    }
    (output_dir / "tracks.json").write_text(json.dumps(payload, indent=2) + "\n")
    if args.save_visualizations:
        save_visualizations(
            frame_paths,
            results,
            output_dir / "visualizations",
            args.marker_radius,
            description,
            output_dir / "overlay_preview.gif" if args.save_overlay_gif else None,
            args.overlay_fps,
        )
    LOGGER.info(
        "Saved results to %s (coverage=%.3f, mask_hit_rate=%s)",
        output_dir,
        payload["coverage"],
        payload["mask_hit_rate"],
    )


if __name__ == "__main__":
    main()
