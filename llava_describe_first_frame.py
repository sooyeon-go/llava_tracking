#!/usr/bin/env python3
"""Per-frame bear localization smoke test for LLaVA-OneVision-2.

Shows each selected RGB frame independently as a single image (not video
tracking), asks where the bear is, and saves natural-language answers plus
optional point overlays.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

LOGGER = logging.getLogger("llava_describe_first_frame")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COORDINATE_SCALE = 1000.0
DEFAULT_PROMPT = (
    "Where is the bear in this image? First briefly describe its location "
    "in natural language (for example: left/center/right, near foreground/"
    "background). Then point to one representative location on the bear "
    "near its center using coordinates scaled from 0 to 1000 relative to "
    "the full image, in this exact format: "
    '<points coords="0 x y">bear</points>'
)


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
        help="Used only to infer the JPEGImages folder when --image/--frames-dir "
        "are omitted.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="RGB frame folder. If omitted, replace Annotations with JPEGImages.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Explicit single image path. Overrides --frames-dir/--max-frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Independently localize the first N frames; 0 keeps every frame.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text prompt for each single-image localization call.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--marker-radius", type=int, default=12)
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
        default=Path("davis_describe_results/bear"),
    )
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    return args


def natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def infer_frames_dir(annotation_dir: Path) -> Path:
    parts = list(annotation_dir.parts)
    try:
        annotation_index = parts.index("Annotations")
    except ValueError as error:
        raise ValueError(
            "--frames-dir or --image is required when --annotation-dir does not "
            "contain an 'Annotations' path component"
        ) from error
    parts[annotation_index] = "JPEGImages"
    return Path(*parts)


def resolve_frame_paths(args: argparse.Namespace) -> list[Path]:
    if args.image is not None:
        image_path = args.image.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        return [image_path]

    frames_dir = (
        args.frames_dir.resolve()
        if args.frames_dir is not None
        else infer_frames_dir(args.annotation_dir.resolve())
    )
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames folder does not exist: {frames_dir}")
    paths = [
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    paths.sort(key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No image files found in: {frames_dir}")
    if args.max_frames == 0 or len(paths) <= args.max_frames:
        return paths
    return paths[: args.max_frames]


def parse_points(text: str) -> list[tuple[float, float]]:
    """Parse Molmo-style <points coords="..."> tags into normalized coords."""
    matches = re.findall(
        r"<points\b[^>]*\bcoords\s*=\s*[\"']([^\"']+)[\"'][^>]*>",
        text,
        flags=re.IGNORECASE,
    )
    points: list[tuple[float, float]] = []
    for coords in matches:
        fields = coords.strip().split()
        values: list[float] = []
        for field in fields:
            try:
                values.append(float(field))
            except ValueError:
                continue
        if len(values) == 2:
            candidates = [(values[0], values[1])]
        elif len(values) >= 3 and len(values) % 3 == 0:
            candidates = [
                (values[index + 1], values[index + 2])
                for index in range(0, len(values), 3)
            ]
        elif len(values) >= 2 and len(values) % 2 == 0:
            candidates = [
                (values[index], values[index + 1])
                for index in range(0, len(values), 2)
            ]
        else:
            continue
        for x, y in candidates:
            if 0 <= x <= COORDINATE_SCALE and 0 <= y <= COORDINATE_SCALE:
                points.append((x, y))
    return points


def normalized_to_pixel(
    point: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    width, height = image_size
    return (
        point[0] / COORDINATE_SCALE * (width - 1),
        point[1] / COORDINATE_SCALE * (height - 1),
    )


def draw_points(
    image: Image.Image,
    points: list[tuple[float, float]],
    radius: int,
) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for point_id, (x, y) in enumerate(points):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline="red",
            width=4,
        )
        draw.line((x - 2 * radius, y, x + 2 * radius, y), fill="red", width=3)
        draw.line((x, y - 2 * radius, x, y + 2 * radius), fill="red", width=3)
        draw.text((x + radius + 4, y - radius), f"pred{point_id}", fill="red")
    return marked


def localize_one_image(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    torch_module,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch_module.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    return processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    frame_paths = resolve_frame_paths(args)
    LOGGER.info(
        "Per-image localization on %d frame(s); not video tracking",
        len(frame_paths),
    )

    dtype = getattr(torch, args.dtype)
    LOGGER.info("Loading processor/model from %s", args.model_path)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    ).eval()

    output_dir = args.output_dir.resolve()
    overlays_dir = output_dir / "overlays"
    raw_dir = output_dir / "raw_outputs"
    if overlays_dir.exists():
        shutil.rmtree(overlays_dir)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(args.prompt + "\n")

    frame_results = []
    for index, image_path in enumerate(frame_paths):
        image = Image.open(image_path).convert("RGB")
        LOGGER.info(
            "[%d/%d] Localizing %s (%dx%d)",
            index + 1,
            len(frame_paths),
            image_path.name,
            *image.size,
        )
        answer = localize_one_image(
            model,
            processor,
            image,
            args.prompt,
            args.max_new_tokens,
            torch,
        )
        normalized_points = parse_points(answer)
        pixel_points = [
            normalized_to_pixel(point, image.size) for point in normalized_points
        ]
        if not normalized_points:
            LOGGER.warning(
                "No <points coords=...> parsed for %s",
                image_path.name,
            )

        (raw_dir / f"{image_path.stem}.txt").write_text(answer + "\n")
        if pixel_points:
            overlay = draw_points(image, pixel_points, args.marker_radius)
        else:
            overlay = image.copy()
        overlay.save(overlays_dir / f"{image_path.stem}.png")

        frame_results.append(
            {
                "frame_index": index,
                "image": str(image_path),
                "image_size": list(image.size),
                "normalized_points": [list(point) for point in normalized_points],
                "pixel_points": [list(point) for point in pixel_points],
                "raw_output": answer,
            }
        )
        print(f"===== {image_path.name} =====")
        print(answer)
        print()

    payload = {
        "mode": "per_image_pointing",
        "prompt": args.prompt,
        "num_frames": len(frame_paths),
        "frames": frame_results,
    }
    (output_dir / "points.json").write_text(json.dumps(payload, indent=2) + "\n")
    # Convenience copies for the first frame.
    if frame_results:
        first = frame_results[0]
        Image.open(frame_paths[0]).convert("RGB").save(
            output_dir / "input_first_frame.png"
        )
        (output_dir / "raw_output.txt").write_text(first["raw_output"] + "\n")
        if first["pixel_points"]:
            Image.open(overlays_dir / f"{frame_paths[0].stem}.png").save(
                output_dir / "point_overlay.png"
            )

    LOGGER.info(
        "Saved per-image localization for %d frames to %s",
        len(frame_paths),
        output_dir,
    )


if __name__ == "__main__":
    main()
