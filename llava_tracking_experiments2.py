#!/usr/bin/env python3
"""Run follow-up sweeps for LLaVA-OV-2 point tracking diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llava_davis_tracking import image_files, infer_frames_dir
from llava_tracking_experiments import (
    Experiment,
    ExperimentModel,
    make_video,
    run_case,
)

LOGGER = logging.getLogger("llava_tracking_experiments2")


@dataclass(frozen=True)
class SweepCase:
    group: str
    folder: str
    name: str
    input_kind: str
    num_frames: int
    max_pixels: int
    max_new_tokens: int
    prompt: str
    sequence: str = "bear"


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
    parser.add_argument("--description", default="the bear")
    parser.add_argument(
        "--motion-annotation-dir",
        type=Path,
        default=Path(
            "/data/shared-vilab/datasets/DAVIS/"
            "Annotations/Full-Resolution/drift-chicane"
        ),
        help=(
            "High-motion DAVIS sequence. The case is recorded as skipped if "
            "this directory or its matching JPEGImages directory is absent."
        ),
    )
    parser.add_argument("--motion-frames-dir", type=Path)
    parser.add_argument("--motion-description", default="a sport car")
    parser.add_argument(
        "--skip-high-motion",
        action="store_true",
        help=(
            "Omit the shared drift-chicane diagnostic. Useful when sweeping "
            "many target sequences to avoid repeating the same control case."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("davis_tracking_experiments2"),
    )
    parser.add_argument("--source-fps", type=float, default=24.0)
    parser.add_argument("--motion-source-fps", type=float, default=24.0)
    parser.add_argument("--overlay-fps", type=float, default=4.0)
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
    args = parser.parse_args()
    if args.source_fps <= 0 or args.motion_source_fps <= 0:
        parser.error("source FPS values must be > 0")
    if args.overlay_fps <= 0:
        parser.error("--overlay-fps must be > 0")
    return args


def tracking_prompt(description: str) -> str:
    expression = description.strip()
    if expression.lower().startswith("track "):
        return expression
    return f"Track {expression}"


def build_sweep_cases(
    description: str,
    motion_description: str,
) -> list[SweepCase]:
    prompt = tracking_prompt(description)
    motion_prompt = tracking_prompt(motion_description)
    cases: list[SweepCase] = []

    # 1) Find the frame-count threshold where trajectories become static.
    for num_frames in (4, 6, 8, 10, 12, 16):
        cases.append(
            SweepCase(
                group="frame_sweep",
                folder=f"{num_frames:02d}_frames",
                name=f"pil_{num_frames}_frames_200704px",
                input_kind="pil_frames",
                num_frames=num_frames,
                max_pixels=200_704,
                max_new_tokens=1024,
                prompt=prompt,
            )
        )

    # 2) Isolate the resolution effect at the successful 8-frame setting.
    for max_pixels in (100_000, 200_704, 400_000, 1_000_000):
        cases.append(
            SweepCase(
                group="resolution_sweep",
                folder=f"{max_pixels:07d}_pixels",
                name=f"pil_8_frames_{max_pixels}px",
                input_kind="pil_frames",
                num_frames=8,
                max_pixels=max_pixels,
                max_new_tokens=1024,
                prompt=prompt,
            )
        )

    # 3) Compare only the media input path at matched settings.
    for input_kind in ("pil_frames", "mp4_path"):
        short_name = "pil" if input_kind == "pil_frames" else "mp4"
        cases.append(
            SweepCase(
                group="input_ab",
                folder=short_name,
                name=f"{short_name}_8_frames_200704px",
                input_kind=input_kind,
                num_frames=8,
                max_pixels=200_704,
                max_new_tokens=1024,
                prompt=prompt,
            )
        )

    # 4) Check whether the model follows a visibly high-motion DAVIS object.
    cases.append(
        SweepCase(
            group="high_motion",
            folder="drift_chicane_8_frames",
            name="high_motion_pil_8_frames_200704px",
            input_kind="pil_frames",
            num_frames=8,
            max_pixels=200_704,
            max_new_tokens=1024,
            prompt=motion_prompt,
            sequence="motion",
        )
    )

    # 5) Test whether 16-frame generation was constrained by output budget.
    cases.append(
        SweepCase(
            group="token_budget",
            folder="16_frames_2048_tokens",
            name="pil_16_frames_200704px_2048_tokens",
            input_kind="pil_frames",
            num_frames=16,
            max_pixels=200_704,
            max_new_tokens=2048,
            prompt=prompt,
        )
    )
    return cases


def resolve_frames(
    annotation_dir: Path,
    frames_dir: Path | None,
) -> tuple[Path, list[Path]]:
    resolved_annotation_dir = annotation_dir.resolve()
    resolved_frames_dir = (
        frames_dir.resolve()
        if frames_dir is not None
        else infer_frames_dir(resolved_annotation_dir)
    )
    return resolved_frames_dir, image_files(resolved_frames_dir)


def case_config(case: SweepCase) -> dict[str, Any]:
    config = asdict(case)
    config["purpose"] = {
        "frame_sweep": "Find the frame-count threshold for static output.",
        "resolution_sweep": "Measure resolution effects with 8 input frames.",
        "input_ab": "Compare PIL-frame and official MP4 processor paths.",
        "high_motion": "Distinguish true tracking from small coordinate jitter.",
        "token_budget": "Test whether 16-frame output needs more decode tokens.",
    }[case.group]
    return config


def run_sweep_case(
    case: SweepCase,
    model: ExperimentModel,
    frame_paths: list[Path],
    video_path: Path,
    source_fps: float,
    case_dir: Path,
    overlay_fps: float,
) -> dict[str, Any]:
    experiment = Experiment(
        number=0,
        name=case.name,
        input_kind=case.input_kind,
        num_frames=case.num_frames,
        max_pixels=case.max_pixels,
        prompt=case.prompt,
        parse_as_tracks=True,
    )
    previous_max_new_tokens = model.max_new_tokens
    model.max_new_tokens = case.max_new_tokens
    try:
        return run_case(
            experiment,
            model,
            frame_paths,
            video_path,
            source_fps,
            case_dir,
            overlay_fps,
        )
    finally:
        model.max_new_tokens = previous_max_new_tokens


def write_summary(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    compact_summary = []
    for result in summary:
        compact_summary.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"raw_output", "processor_debug"}
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(compact_summary, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    _, bear_frames = resolve_frames(args.annotation_dir, args.frames_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_frames: list[Path] | None = None
    motion_resolution_error: str | None = None
    try:
        _, motion_frames = resolve_frames(
            args.motion_annotation_dir,
            args.motion_frames_dir,
        )
    except Exception:
        motion_resolution_error = traceback.format_exc()
        LOGGER.warning(
            "High-motion sequence unavailable; its case will be skipped"
        )

    cases = build_sweep_cases(
        args.description,
        args.motion_description,
    )
    if args.skip_high_motion:
        cases = [case for case in cases if case.group != "high_motion"]
    (output_dir / "experiment_plan.json").write_text(
        json.dumps([case_config(case) for case in cases], indent=2) + "\n"
    )

    video_dir = output_dir / "_inputs"
    video_dir.mkdir(exist_ok=True)
    bear_video_path = video_dir / "bear.mp4"
    bear_video_error: str | None = None
    try:
        make_video(bear_frames, bear_video_path, args.source_fps)
    except Exception:
        bear_video_error = traceback.format_exc()
        LOGGER.exception("Could not create bear MP4")

    model = ExperimentModel(args)
    summary: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        case_dir = output_dir / case.group / case.folder
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True)
        (case_dir / "config.json").write_text(
            json.dumps(case_config(case), indent=2) + "\n"
        )
        LOGGER.info(
            "Case %d/%d: %s",
            case_index,
            len(cases),
            case.name,
        )

        if case.sequence == "motion" and motion_frames is None:
            error_text = (
                "Skipped because the high-motion sequence is unavailable:\n"
                + (motion_resolution_error or "unknown error")
            )
            (case_dir / "skipped.txt").write_text(error_text)
            case_result: dict[str, Any] = {
                "status": "skipped",
                "error": error_text,
            }
        elif case.input_kind == "mp4_path" and bear_video_error is not None:
            error_text = "MP4 preparation failed:\n" + bear_video_error
            (case_dir / "error.txt").write_text(error_text)
            case_result = {"status": "failed", "error": error_text}
        else:
            selected_frames = (
                motion_frames if case.sequence == "motion" else bear_frames
            )
            selected_video = bear_video_path
            selected_fps = (
                args.motion_source_fps
                if case.sequence == "motion"
                else args.source_fps
            )
            try:
                case_result = run_sweep_case(
                    case,
                    model,
                    selected_frames,
                    selected_video,
                    selected_fps,
                    case_dir,
                    args.overlay_fps,
                )
            except Exception:
                error_text = traceback.format_exc()
                (case_dir / "error.txt").write_text(error_text)
                LOGGER.exception("Case failed; continuing: %s", case.name)
                model.recover_after_failure()
                case_result = {"status": "failed", "error": error_text}

        summary.append(
            {
                "group": case.group,
                "folder": case.folder,
                "name": case.name,
                **case_result,
            }
        )
        write_summary(output_dir, summary)

    LOGGER.info("Finished %d follow-up cases: %s", len(cases), output_dir)


if __name__ == "__main__":
    main()
