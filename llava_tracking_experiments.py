#!/usr/bin/env python3
"""Run five LLaVA-OV-2 tracking diagnostics on one DAVIS sequence."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from llava_davis_tracking import (
    image_files,
    infer_frames_dir,
    parse_tracks,
    select_frames,
)

LOGGER = logging.getLogger("llava_tracking_experiments")
COORDINATE_SCALE = 1000.0


@dataclass(frozen=True)
class Experiment:
    number: int
    name: str
    input_kind: str
    num_frames: int
    max_pixels: int
    prompt: str
    parse_as_tracks: bool = True


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
    )
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("davis_tracking_experiments/bear"),
    )
    parser.add_argument("--description", default="the bear")
    parser.add_argument("--source-fps", type=float, default=24.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--overlay-fps", type=float, default=4.0)
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
    args = parser.parse_args()
    if args.source_fps <= 0:
        parser.error("--source-fps must be > 0")
    if args.overlay_fps <= 0:
        parser.error("--overlay-fps must be > 0")
    return args


def build_experiments(description: str) -> list[Experiment]:
    track_prompt = (
        description.strip()
        if description.strip().lower().startswith("track ")
        else f"Track {description.strip()}"
    )
    return [
        Experiment(
            1,
            "pil_16_official_resolution",
            "pil_frames",
            16,
            200_704,
            track_prompt,
        ),
        Experiment(
            2,
            "pil_8_official_resolution",
            "pil_frames",
            8,
            200_704,
            track_prompt,
        ),
        Experiment(
            3,
            "pil_16_high_resolution",
            "pil_frames",
            16,
            4_000_000,
            track_prompt,
        ),
        Experiment(
            4,
            "mp4_16_official_input",
            "mp4_path",
            16,
            200_704,
            track_prompt,
        ),
        Experiment(
            5,
            "mp4_16_motion_qa",
            "mp4_path",
            16,
            200_704,
            (
                f"Describe how {description.strip()} moves through this video. "
                "Mention its starting location, ending location, and direction "
                "of motion."
            ),
            parse_as_tracks=False,
        ),
    ]


def make_video(
    frame_paths: list[Path],
    output_path: Path,
    source_fps: float,
) -> None:
    """Encode the image sequence so cases 4/5 use the official MP4 path."""
    manifest_path = output_path.with_suffix(".ffconcat")
    duration = 1.0 / source_fps
    lines = ["ffconcat version 1.0"]
    for frame_path in frame_paths:
        escaped_path = str(frame_path.resolve()).replace("'", r"'\''")
        lines.append(f"file '{escaped_path}'")
        lines.append(f"duration {duration:.12f}")
    # The concat demuxer needs the final file repeated to honor its duration.
    escaped_last = str(frame_paths[-1].resolve()).replace("'", r"'\''")
    lines.append(f"file '{escaped_last}'")
    manifest_path.write_text("\n".join(lines) + "\n")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-safe",
        "0",
        "-f",
        "concat",
        "-i",
        str(manifest_path),
        "-r",
        str(source_fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)


class ExperimentModel:
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

    def generate(
        self,
        media: list[Image.Image] | Path,
        prompt: str,
        max_pixels: int,
        num_frames: int,
        max_new_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self.processor.video_processor.max_pixels = max_pixels
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
        processor_kwargs: dict[str, Any] = {
            "text": [chat_text],
            "padding": True,
            "return_tensors": "pt",
        }
        if isinstance(media, Path):
            processor_kwargs["videos"] = [str(media)]
            processor_kwargs["num_frames"] = num_frames
        else:
            processor_kwargs["videos"] = media
        inputs = self.processor(**processor_kwargs)
        debug_info = {
            "input_ids_shape": list(inputs["input_ids"].shape),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "patch_positions_shape": list(inputs["patch_positions"].shape),
        }
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        tokenizer = self.processor.tokenizer
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        answer = self.processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return answer, debug_info

    def recover_after_failure(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def normalized_to_pixel(
    point: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    width, height = image_size
    return (
        point[0] / COORDINATE_SCALE * (width - 1),
        point[1] / COORDINATE_SCALE * (height - 1),
    )


def closest_track_timestamp(
    tracks: dict[float, dict[int, tuple[float, float]]],
    expected_timestamp: float,
    tolerance: float,
) -> float | None:
    if not tracks:
        return None
    closest = min(
        tracks,
        key=lambda timestamp: abs(timestamp - expected_timestamp),
    )
    if abs(closest - expected_timestamp) > tolerance:
        return None
    return closest


def save_overlay_gif(
    frames: list[Image.Image],
    gif_path: Path,
    overlay_fps: float,
) -> None:
    if not frames:
        return
    previews: list[Image.Image] = []
    for frame in frames:
        preview = frame.copy().convert("RGB")
        if preview.width > 832:
            preview.thumbnail((832, 832), Image.Resampling.LANCZOS)
        previews.append(preview)
    duration_ms = max(1, round(1000 / overlay_fps))
    previews[0].save(
        gif_path,
        save_all=True,
        append_images=previews[1:],
        duration=duration_ms,
        loop=0,
    )


def save_track_outputs(
    case_dir: Path,
    tracks: dict[float, dict[int, tuple[float, float]]],
    frame_paths: list[Path],
    expected_timestamps: list[float],
    overlay_fps: float,
) -> dict[str, Any]:
    overlays_dir = case_dir / "overlays"
    if overlays_dir.exists():
        shutil.rmtree(overlays_dir)
    overlays_dir.mkdir(parents=True)

    timestamp_gaps = [
        later - earlier
        for earlier, later in zip(
            expected_timestamps,
            expected_timestamps[1:],
        )
        if later > earlier
    ]
    timestamp_tolerance = (
        min(timestamp_gaps) / 2.0 + 1e-6 if timestamp_gaps else 0.051
    )
    frame_results: list[dict[str, Any]] = []
    all_pixel_points: list[tuple[float, float]] = []
    gif_frames: list[Image.Image] = []
    for frame_index, (frame_path, expected_timestamp) in enumerate(
        zip(frame_paths, expected_timestamps)
    ):
        matched_timestamp = closest_track_timestamp(
            tracks,
            expected_timestamp,
            timestamp_tolerance,
        )
        frame_tracks = (
            tracks.get(matched_timestamp, {})
            if matched_timestamp is not None
            else {}
        )
        image = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        points = []
        for point_id, normalized_point in sorted(frame_tracks.items()):
            pixel_point = normalized_to_pixel(normalized_point, image.size)
            all_pixel_points.append(pixel_point)
            x, y = pixel_point
            radius = 12
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline="red",
                width=4,
            )
            draw.line((x - radius, y, x + radius, y), fill="red", width=3)
            draw.line((x, y - radius, x, y + radius), fill="red", width=3)
            draw.text((x + radius + 3, y - radius), f"pred{point_id}", fill="red")
            points.append(
                {
                    "point_id": point_id,
                    "normalized_xy": list(normalized_point),
                    "pixel_xy": list(pixel_point),
                }
            )
        image.save(overlays_dir / f"{frame_path.stem}.png")
        gif_frames.append(image)
        frame_results.append(
            {
                "frame_index": frame_index,
                "frame_path": str(frame_path),
                "expected_timestamp": expected_timestamp,
                "matched_output_timestamp": matched_timestamp,
                "points": points,
            }
        )

    save_overlay_gif(gif_frames, case_dir / "overlay_preview.gif", overlay_fps)
    span_px = None
    if all_pixel_points:
        xs = [point[0] for point in all_pixel_points]
        ys = [point[1] for point in all_pixel_points]
        span_px = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    payload = {
        "parsed_tracks": {
            str(timestamp): {
                str(point_id): list(point)
                for point_id, point in frame_tracks.items()
            }
            for timestamp, frame_tracks in tracks.items()
        },
        "trajectory_span_px": span_px,
        "frames": frame_results,
    }
    (case_dir / "tracks.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run_case(
    experiment: Experiment,
    model: ExperimentModel,
    all_frame_paths: list[Path],
    video_path: Path,
    source_fps: float,
    case_dir: Path,
    overlay_fps: float,
) -> dict[str, Any]:
    frame_paths, original_indices = select_frames(
        all_frame_paths,
        experiment.num_frames,
        "uniform",
    )
    if experiment.input_kind == "pil_frames":
        media: list[Image.Image] | Path = [
            Image.open(path).convert("RGB") for path in frame_paths
        ]
        expected_timestamps = [float(index) for index in range(len(frame_paths))]
    else:
        media = video_path
        expected_timestamps = [
            round(index / source_fps, 1) for index in original_indices
        ]

    raw_output, processor_debug = model.generate(
        media,
        experiment.prompt,
        experiment.max_pixels,
        experiment.num_frames,
        max_new_tokens=512 if not experiment.parse_as_tracks else None,
    )
    (case_dir / "prompt.txt").write_text(experiment.prompt + "\n")
    (case_dir / "raw_output.txt").write_text(raw_output + "\n")
    (case_dir / "processor_debug.json").write_text(
        json.dumps(processor_debug, indent=2) + "\n"
    )

    result: dict[str, Any] = {
        "status": "completed",
        "raw_output": raw_output,
        "processor_debug": processor_debug,
    }
    if experiment.parse_as_tracks:
        tracks = parse_tracks(raw_output)
        track_payload = save_track_outputs(
            case_dir,
            tracks,
            frame_paths,
            expected_timestamps,
            overlay_fps,
        )
        result["num_parsed_timestamps"] = len(tracks)
        result["trajectory_span_px"] = track_payload["trajectory_span_px"]
    else:
        # Case 5 has no points; still save an input GIF for side-by-side review.
        input_frames = [Image.open(path).convert("RGB") for path in frame_paths]
        save_overlay_gif(input_frames, case_dir / "input_preview.gif", overlay_fps)
        result["gif"] = "input_preview.gif"
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    annotation_dir = args.annotation_dir.resolve()
    frames_dir = (
        args.frames_dir.resolve()
        if args.frames_dir is not None
        else infer_frames_dir(annotation_dir)
    )
    all_frame_paths = image_files(frames_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments(args.description)
    (output_dir / "experiment_plan.json").write_text(
        json.dumps([asdict(experiment) for experiment in experiments], indent=2)
        + "\n"
    )

    video_path = output_dir / "input_sequence.mp4"
    video_error: str | None = None
    try:
        LOGGER.info("Encoding image sequence for official MP4-input cases")
        make_video(all_frame_paths, video_path, args.source_fps)
    except Exception:
        video_error = traceback.format_exc()
        LOGGER.exception("Could not create MP4; cases 4 and 5 will record errors")

    model = ExperimentModel(args)
    summary = []
    for experiment in experiments:
        case_dir = output_dir / str(experiment.number)
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True)
        (case_dir / "config.json").write_text(
            json.dumps(asdict(experiment), indent=2) + "\n"
        )
        LOGGER.info(
            "Case %d/5: %s",
            experiment.number,
            experiment.name,
        )
        if experiment.input_kind == "mp4_path" and video_error is not None:
            error_text = "MP4 preparation failed:\n" + video_error
            (case_dir / "error.txt").write_text(error_text)
            case_result = {"status": "failed", "error": error_text}
        else:
            try:
                case_result = run_case(
                    experiment,
                    model,
                    all_frame_paths,
                    video_path,
                    args.source_fps,
                    case_dir,
                    args.overlay_fps,
                )
            except Exception:
                error_text = traceback.format_exc()
                (case_dir / "error.txt").write_text(error_text)
                LOGGER.exception("Case %d failed; continuing", experiment.number)
                model.recover_after_failure()
                case_result = {"status": "failed", "error": error_text}
        summary.append(
            {
                "number": experiment.number,
                "name": experiment.name,
                **case_result,
            }
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    LOGGER.info("Finished all five cases: %s", output_dir)


if __name__ == "__main__":
    main()
