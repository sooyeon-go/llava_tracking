#!/usr/bin/env python3
"""Single-image bear localization smoke test for LLaVA-OneVision-2.

Loads the first DAVIS RGB frame, asks where the bear is (natural-language
location + a point), and saves the raw answer plus an optional overlay.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
        help="Explicit first-frame image path. Overrides --frames-dir.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text prompt for the single-image localization demo.",
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
    return parser.parse_args()


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


def first_frame_path(args: argparse.Namespace) -> Path:
    if args.image is not None:
        image_path = args.image.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        return image_path

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
    return paths[0]


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
        # Common formats:
        #   "x y"
        #   "0 x y"  (id, x, y)
        #   "0 x y 1 x y ..."
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    image_path = first_frame_path(args)
    image = Image.open(image_path).convert("RGB")
    LOGGER.info("Localizing bear in first frame: %s (%dx%d)", image_path, *image.size)

    dtype = getattr(torch, args.dtype)
    LOGGER.info("Loading processor/model from %s", args.model_path)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    # HF demo uses AutoModelForVision2Seq; newer transformers prefer
    # AutoModelForImageTextToText for the same OV-2 checkpoint.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    ).eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
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
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    answer = processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    normalized_points = parse_points(answer)
    pixel_points = [
        normalized_to_pixel(point, image.size) for point in normalized_points
    ]
    if not normalized_points:
        LOGGER.warning(
            "No <points coords=...> could be parsed from model output; "
            "check raw_output.txt"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "input_first_frame.png")
    (output_dir / "prompt.txt").write_text(args.prompt + "\n")
    (output_dir / "raw_output.txt").write_text(answer + "\n")
    payload = {
        "image": str(image_path),
        "image_size": list(image.size),
        "prompt": args.prompt,
        "normalized_points": [list(point) for point in normalized_points],
        "pixel_points": [list(point) for point in pixel_points],
        "raw_output": answer,
    }
    (output_dir / "points.json").write_text(json.dumps(payload, indent=2) + "\n")
    if pixel_points:
        overlay = draw_points(image, pixel_points, args.marker_radius)
        overlay.save(output_dir / "point_overlay.png")
        LOGGER.info("Parsed %d point(s); wrote point_overlay.png", len(pixel_points))

    print(answer)
    LOGGER.info("Saved localization results to %s", output_dir)


if __name__ == "__main__":
    main()
