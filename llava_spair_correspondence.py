#!/usr/bin/env python3
"""Evaluate LLaVA-OneVision-2 on SPair-71k point correspondence.

The source keypoint is shown as a red cross and also provided as a normalized
coordinate. The model predicts the corresponding point in the target image.
Predictions are written incrementally to JSONL so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image, ImageDraw

LOGGER = logging.getLogger("llava_spair_correspondence")
COORDINATE_SCALE = 1000.0
DEFAULT_MIN_INPUT_PIXELS = 832 * 480
VISION_ALIGNMENT = 28  # patch_size (14) * spatial_merge_size (2)


@dataclass(frozen=True)
class PredictionKey:
    pair_filename: str
    keypoint_index: int

    @property
    def value(self) -> str:
        return f"{self.pair_filename}::{self.keypoint_index}"


@dataclass
class Prediction:
    pair_filename: str
    pair_id: int
    split: str
    category: str
    source_image: str
    target_image: str
    keypoint_index: int
    keypoint_id: str
    source_point: list[float]
    target_ground_truth: list[float]
    target_prediction: list[float] | None
    source_input_size: list[int]
    target_input_size: list[int]
    target_visible: bool | None
    valid_prediction: bool
    pixel_error: float | None
    pck_threshold: float
    pck_correct: bool
    raw_output: str
    parse_error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLaVA-OneVision-2 correspondence on SPair-71k."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/data/shared-vilab/pretrained_models/VLM_models/"
            "LLaVA-OneVision-2-8B-Instruct"
        ),
        help="Local LLaVA-OneVision-2 checkpoint directory.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/shared-vilab/datasets/spair-71k/SPair-71k"),
    )
    parser.add_argument("--split", choices=("trn", "val", "test"), default="test")
    parser.add_argument(
        "--layout-size", choices=("small", "large"), default="large"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=1,
        help="Maximum pairs to process; 0 processes the entire split.",
    )
    parser.add_argument(
        "--pair-sampling",
        choices=("first", "stratified"),
        default="stratified",
        help=(
            "How to select --max-pairs. stratified round-robins across object "
            "categories so a short test is not dominated by one category."
        ),
    )
    parser.add_argument(
        "--category",
        action="append",
        help="Category filter; repeat for multiple categories.",
    )
    parser.add_argument(
        "--keypoint-index",
        type=int,
        help="Only process this local shared-keypoint index.",
    )
    parser.add_argument("--pck-alpha", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--prompt-format",
        choices=("native-track", "json"),
        default="native-track",
        help="Use OneVision2's pretrained track grammar or generic JSON.",
    )
    parser.add_argument(
        "--min-input-pixels",
        type=int,
        default=DEFAULT_MIN_INPUT_PIXELS,
        help=(
            "Upscale each image to at least this pixel area while preserving "
            "aspect ratio (default: 832*480). Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device_map, e.g. "auto" or "cuda:0".',
    )
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
        "--marker-radius",
        type=int,
        default=10,
        help="Source marker radius in pixels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("llava_correspondence_results"),
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save source/target images with prediction and ground truth.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing results JSONL instead of resuming.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare marked inputs and prompts without loading a model.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the selected pairs across this many workers.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Zero-based worker index in [0, --num-shards).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only rebuild summary JSON from an existing predictions JSONL.",
    )
    args = parser.parse_args()

    if args.max_pairs < 0:
        parser.error("--max-pairs must be >= 0")
    if args.pck_alpha <= 0:
        parser.error("--pck-alpha must be > 0")
    if args.min_input_pixels < 0:
        parser.error("--min-input-pixels must be >= 0")
    if args.num_shards <= 0:
        parser.error("--num-shards must be > 0")
    if not (0 <= args.shard_id < args.num_shards):
        parser.error("--shard-id must satisfy 0 <= shard-id < num-shards")
    return args


def shard_pair_ids(pair_ids: list[str], num_shards: int, shard_id: int) -> list[str]:
    if num_shards == 1:
        return pair_ids
    return [
        pair_id
        for index, pair_id in enumerate(pair_ids)
        if index % num_shards == shard_id
    ]


def read_pair_ids(
    root: Path,
    split: str,
    layout_size: str,
    categories: set[str] | None,
) -> list[str]:
    layout_path = root / "Layout" / layout_size / f"{split}.txt"
    if not layout_path.is_file():
        raise FileNotFoundError(f"SPair layout file not found: {layout_path}")

    pair_ids = [line.strip() for line in layout_path.read_text().splitlines()]
    pair_ids = [pair_id for pair_id in pair_ids if pair_id]
    if categories:
        pair_ids = [
            pair_id
            for pair_id in pair_ids
            if pair_id.rsplit(":", maxsplit=1)[-1] in categories
        ]
    return pair_ids


def sample_pair_ids(
    pair_ids: list[str],
    max_pairs: int,
    sampling: str,
) -> list[str]:
    if max_pairs == 0 or len(pair_ids) <= max_pairs:
        return pair_ids
    if sampling == "first":
        return pair_ids[:max_pairs]

    by_category: dict[str, list[str]] = {}
    for pair_id in pair_ids:
        category = pair_id.rsplit(":", maxsplit=1)[-1]
        by_category.setdefault(category, []).append(pair_id)

    selected = []
    categories = sorted(by_category)
    pair_index = 0
    while len(selected) < max_pairs:
        added = False
        for category in categories:
            category_pairs = by_category[category]
            if pair_index < len(category_pairs):
                selected.append(category_pairs[pair_index])
                added = True
                if len(selected) == max_pairs:
                    break
        if not added:
            break
        pair_index += 1
    return selected


def load_annotation(root: Path, split: str, pair_filename: str) -> dict[str, Any]:
    annotation_path = root / "PairAnnotation" / split / f"{pair_filename}.json"
    with annotation_path.open() as annotation_file:
        annotation = json.load(annotation_file)
    if len(annotation["src_kps"]) != len(annotation["trg_kps"]):
        raise ValueError(f"Mismatched keypoint counts: {annotation_path}")
    return annotation


def draw_point_marker(
    image: Image.Image,
    point: Iterable[float],
    radius: int,
    color: str = "red",
) -> Image.Image:
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    x, y = (round(float(value)) for value in point)
    line_width = max(2, radius // 3)
    outline_width = line_width + 2
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline="white",
        width=outline_width,
    )
    draw.line(
        (x - radius * 2, y, x + radius * 2, y),
        fill="white",
        width=outline_width,
    )
    draw.line(
        (x, y - radius * 2, x, y + radius * 2),
        fill="white",
        width=outline_width,
    )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=color,
        width=line_width,
    )
    draw.line(
        (x - radius * 2, y, x + radius * 2, y),
        fill=color,
        width=line_width,
    )
    draw.line(
        (x, y - radius * 2, x, y + radius * 2),
        fill=color,
        width=line_width,
    )
    return marked


def prepare_input_image(
    image: Image.Image,
    min_pixels: int,
    alignment: int = VISION_ALIGNMENT,
) -> tuple[Image.Image, tuple[float, float]]:
    """Upscale to a minimum token budget and align dimensions to vision patches."""
    image = image.convert("RGB")
    width, height = image.size
    if min_pixels <= 0 or width * height >= min_pixels:
        return image.copy(), (1.0, 1.0)

    scale = math.sqrt(min_pixels / (width * height))
    resized_width = math.ceil(width * scale / alignment) * alignment
    resized_height = math.ceil(height * scale / alignment) * alignment
    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    return resized, (resized_width / width, resized_height / height)


def scale_point(
    point: Iterable[float],
    scale: tuple[float, float],
) -> tuple[float, float]:
    x, y = (float(value) for value in point)
    return x * scale[0], y * scale[1]


def normalized_point(point: Iterable[float], image_size: tuple[int, int]) -> tuple[int, int]:
    x, y = (float(value) for value in point)
    width, height = image_size
    return (
        round(x / max(width - 1, 1) * COORDINATE_SCALE),
        round(y / max(height - 1, 1) * COORDINATE_SCALE),
    )


def build_prompt(
    category: str,
    source_point: Iterable[float],
    source_size: tuple[int, int],
    prompt_format: str,
) -> str:
    source_x, source_y = normalized_point(source_point, source_size)
    task = (
        "You are solving semantic point correspondence between two different "
        f"instances of the same object category ({category}). Image 1 is the "
        "SOURCE frame at timestamp 0.0 and Image 2 is the TARGET frame at "
        "timestamp 1.0. In Image 1, the query point is the "
        f"center of the red cross, at normalized coordinate ({source_x}, "
        f"{source_y}) on a 0-{int(COORDINATE_SCALE)} scale. Find the exact "
        "semantically corresponding anatomical or structural point in Image 2. "
        "Account for viewpoint, articulation, scale, and deformation. Do not "
        "match the red color itself. Coordinates refer to the full target image, "
        f"with (0, 0) at top-left and ({int(COORDINATE_SCALE)}, "
        f"{int(COORDINATE_SCALE)}) at bottom-right. "
    )
    if prompt_format == "native-track":
        source_track = (
            f'<tracks coords="0.0 0 {source_x} {source_y}">'
            "source point</tracks>"
        )
        output_example = (
            f'<tracks coords="0.0 0 {source_x} {source_y};'
            '1.0 0 <target_x> <target_y>">source point</tracks>'
        )
        return (
            f"{task}\nThe source point is:\n{source_track}\n"
            "Return its absolute position in both frames using OneVision2's "
            "native frame-major track grammar. Use point id 0 and integer "
            "coordinates from 0 to 1000. Answer with exactly one track and no "
            f"explanation:\n{output_example}"
        )
    return (
        f"{task}If the corresponding point is visible, answer only as "
        '{"x": <0-1000>, "y": <0-1000>, "visible": true}. '
        "If it is definitely outside the image or fully occluded, answer only as "
        '{"x": null, "y": null, "visible": false}.'
    )


def _native_track_prediction(text: str) -> tuple[float, float, bool] | None:
    """Read the final frame's point-id-0 coordinate from native track text."""
    track_matches = re.findall(
        r"<tracks\b[^>]*\bcoords\s*=\s*[\"']([^\"']+)[\"'][^>]*>",
        text,
        flags=re.IGNORECASE,
    )
    if not track_matches:
        return None

    frames: list[tuple[float, float, float]] = []
    for frame_group in track_matches[-1].split(";"):
        fields = frame_group.strip().split()
        if len(fields) < 4:
            continue
        try:
            timestamp = float(fields[0])
            point_fields = fields[1:]
            for index in range(0, len(point_fields) - 2, 3):
                point_id = int(float(point_fields[index]))
                x = float(point_fields[index + 1])
                y = float(point_fields[index + 2])
                if point_id == 0:
                    frames.append((timestamp, x, y))
                    break
        except ValueError:
            continue
    if not frames:
        return None
    _, x, y = max(frames, key=lambda item: item[0])
    return x, y, True


def _json_prediction(text: str) -> tuple[float, float, bool] | None:
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            value = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        visible = value.get("visible", True)
        if visible is False:
            return math.nan, math.nan, False
        if value.get("x") is not None and value.get("y") is not None:
            return float(value["x"]), float(value["y"]), bool(visible)
    return None


def parse_prediction(text: str) -> tuple[float | None, float | None, bool | None]:
    """Parse JSON, XML-like point tags, or a final coordinate pair."""
    native_track = _native_track_prediction(text)
    if native_track is not None:
        return native_track

    parsed = _json_prediction(text)
    if parsed is not None:
        x, y, visible = parsed
        if not visible:
            return None, None, False
        return x, y, True

    x_match = re.search(
        r"""(?:\bx\b|x_coord(?:inate)?)\s*["'=:\s]+\s*(-?\d+(?:\.\d+)?)""",
        text,
        flags=re.IGNORECASE,
    )
    y_match = re.search(
        r"""(?:\by\b|y_coord(?:inate)?)\s*["'=:\s]+\s*(-?\d+(?:\.\d+)?)""",
        text,
        flags=re.IGNORECASE,
    )
    if x_match and y_match:
        return float(x_match.group(1)), float(y_match.group(1)), True

    pairs = re.findall(
        r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*[,;]\s*(-?\d+(?:\.\d+)?)\s*[\)\]]",
        text,
    )
    if pairs:
        x, y = pairs[-1]
        return float(x), float(y), True
    return None, None, None


def coordinate_to_pixel(
    x: float,
    y: float,
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    # Accept [0, 1] fractions as a fallback, but prefer the requested [0, 1000].
    scale = 1.0 if max(abs(x), abs(y)) <= 1.0 else COORDINATE_SCALE
    normalized_x, normalized_y = x / scale, y / scale
    if not (0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0):
        return None
    width, height = image_size
    return normalized_x * (width - 1), normalized_y * (height - 1)


class LlavaOneVisionPredictor:
    def __init__(
        self,
        model_path: Path,
        dtype_name: str,
        device_map: str,
        attn_implementation: str,
        max_new_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        dtype = getattr(torch, dtype_name)
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": dtype,
            "device_map": device_map,
            "attn_implementation": attn_implementation,
        }

        LOGGER.info("Loading processor from %s", model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        LOGGER.info("Loading model from %s", model_path)
        self.model = self._load_model(model_path, model_kwargs).eval()
        self.dtype = dtype

    @staticmethod
    def _load_model(model_path: Path, model_kwargs: dict[str, Any]) -> Any:
        from transformers import AutoModelForCausalLM

        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(
                model_path, **model_kwargs
            )
        except (ValueError, KeyError):
            LOGGER.info("Falling back to AutoModelForCausalLM")
            return AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    def _input_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

    def predict(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        prompt: str,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
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
            images=[source_image, target_image],
            padding=True,
            return_tensors="pt",
        )
        device = self._input_device()
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        new_ids = generated_ids[:, prompt_length:]
        return self.processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def pck_threshold(annotation: dict[str, Any], alpha: float) -> float:
    x_min, y_min, x_max, y_max = annotation["trg_bndbox"]
    return max(float(x_max - x_min), float(y_max - y_min)) * alpha


def point_error(
    prediction: tuple[float, float] | None,
    ground_truth: Iterable[float],
) -> float | None:
    if prediction is None:
        return None
    gt_x, gt_y = (float(value) for value in ground_truth)
    return math.hypot(prediction[0] - gt_x, prediction[1] - gt_y)


def prediction_from_output(
    annotation: dict[str, Any],
    split: str,
    keypoint_index: int,
    target_size: tuple[int, int],
    source_input_size: tuple[int, int],
    target_input_size: tuple[int, int],
    raw_output: str,
    alpha: float,
) -> Prediction:
    normalized_x, normalized_y, visible = parse_prediction(raw_output)
    predicted_point = None
    parse_error = None
    if visible is True and normalized_x is not None and normalized_y is not None:
        predicted_point = coordinate_to_pixel(
            normalized_x, normalized_y, target_size
        )
        if predicted_point is None:
            parse_error = "Predicted coordinates are outside [0, 1000]"
    elif visible is False:
        parse_error = "Model predicted that the point is not visible"
    else:
        parse_error = "Could not parse target coordinates"

    ground_truth = annotation["trg_kps"][keypoint_index]
    error = point_error(predicted_point, ground_truth)
    threshold = pck_threshold(annotation, alpha)
    keypoint_ids = annotation.get("kps_ids", [])
    keypoint_id = (
        str(keypoint_ids[keypoint_index])
        if keypoint_index < len(keypoint_ids)
        else str(keypoint_index)
    )
    return Prediction(
        pair_filename=annotation["filename"],
        pair_id=int(annotation["pair_id"]),
        split=split,
        category=annotation["category"],
        source_image=annotation["src_imname"],
        target_image=annotation["trg_imname"],
        keypoint_index=keypoint_index,
        keypoint_id=keypoint_id,
        source_point=list(map(float, annotation["src_kps"][keypoint_index])),
        target_ground_truth=list(map(float, ground_truth)),
        target_prediction=(
            [float(predicted_point[0]), float(predicted_point[1])]
            if predicted_point is not None
            else None
        ),
        source_input_size=list(source_input_size),
        target_input_size=list(target_input_size),
        target_visible=visible,
        valid_prediction=predicted_point is not None,
        pixel_error=error,
        pck_threshold=threshold,
        pck_correct=error is not None and error <= threshold,
        raw_output=raw_output,
        parse_error=parse_error,
    )


def read_completed_keys(results_path: Path) -> set[str]:
    completed: set[str] = set()
    if not results_path.is_file():
        return completed
    with results_path.open() as results_file:
        for line_number, line in enumerate(results_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                completed.add(
                    PredictionKey(
                        record["pair_filename"], int(record["keypoint_index"])
                    ).value
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                LOGGER.warning(
                    "Ignoring invalid line %d in %s", line_number, results_path
                )
    return completed


def selected_keypoints(
    annotation: dict[str, Any],
    requested_index: int | None,
) -> Iterator[int]:
    count = len(annotation["src_kps"])
    if requested_index is None:
        yield from range(count)
    elif 0 <= requested_index < count:
        yield requested_index


def draw_evaluation_visualization(
    source_image: Image.Image,
    target_image: Image.Image,
    prediction: Prediction,
    marker_radius: int,
) -> Image.Image:
    source = draw_point_marker(
        source_image, prediction.source_point, marker_radius, color="red"
    )
    target = target_image.convert("RGB").copy()
    target_draw = ImageDraw.Draw(target)
    gt_x, gt_y = prediction.target_ground_truth
    target_draw.ellipse(
        (
            gt_x - marker_radius,
            gt_y - marker_radius,
            gt_x + marker_radius,
            gt_y + marker_radius,
        ),
        outline="blue",
        width=3,
    )
    if prediction.target_prediction is not None:
        pred_x, pred_y = prediction.target_prediction
        target_draw.line(
            (
                pred_x - marker_radius,
                pred_y,
                pred_x + marker_radius,
                pred_y,
            ),
            fill="lime",
            width=4,
        )
        target_draw.line(
            (
                pred_x,
                pred_y - marker_radius,
                pred_x,
                pred_y + marker_radius,
            ),
            fill="lime",
            width=4,
        )

    canvas = Image.new(
        "RGB",
        (source.width + target.width, max(source.height, target.height)),
        color="black",
    )
    canvas.paste(source, (0, 0))
    canvas.paste(target, (source.width, 0))
    return canvas


def write_summary(results_path: Path, summary_path: Path, args: argparse.Namespace) -> None:
    records = []
    if results_path.is_file():
        with results_path.open() as results_file:
            for line in results_file:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    valid = [record for record in records if record.get("valid_prediction")]
    correct = [record for record in records if record.get("pck_correct")]
    errors = [
        float(record["pixel_error"])
        for record in valid
        if record.get("pixel_error") is not None
    ]
    category_summary = {}
    categories = sorted(
        {
            str(record.get("category"))
            for record in records
            if record.get("category") is not None
        }
    )
    for category in categories:
        category_records = [
            record for record in records if record.get("category") == category
        ]
        category_valid = [
            record
            for record in category_records
            if record.get("valid_prediction")
        ]
        category_correct = [
            record
            for record in category_records
            if record.get("pck_correct")
        ]
        category_summary[category] = {
            "num_predictions": len(category_records),
            "num_valid_predictions": len(category_valid),
            "parse_rate": (
                len(category_valid) / len(category_records)
                if category_records
                else 0.0
            ),
            "pck": (
                len(category_correct) / len(category_records)
                if category_records
                else 0.0
            ),
            "pck_among_valid": (
                len(category_correct) / len(category_valid)
                if category_valid
                else 0.0
            ),
        }
    summary = {
        "split": args.split,
        "layout_size": args.layout_size,
        "pair_sampling": args.pair_sampling,
        "pck_alpha": args.pck_alpha,
        "prompt_format": args.prompt_format,
        "min_input_pixels": args.min_input_pixels,
        "num_predictions": len(records),
        "num_valid_predictions": len(valid),
        "parse_rate": len(valid) / len(records) if records else 0.0,
        "pck": len(correct) / len(records) if records else 0.0,
        "pck_among_valid": len(correct) / len(valid) if valid else 0.0,
        "mean_pixel_error_valid": sum(errors) / len(errors) if errors else None,
        "by_category": category_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Summary: %s", json.dumps(summary))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards > 1:
        results_path = (
            output_dir
            / "shards"
            / f"shard{args.shard_id:02d}"
            / f"{args.split}_predictions.jsonl"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_path = output_dir / f"{args.split}_predictions.jsonl"
    summary_path = output_dir / f"{args.split}_summary.json"
    visualization_dir = output_dir / "visualizations" / args.split
    dry_run_dir = output_dir / "dry_run"

    if args.summary_only:
        write_summary(results_path if args.num_shards == 1 else (
            output_dir / f"{args.split}_predictions.jsonl"
        ), summary_path, args)
        return

    if args.overwrite and results_path.exists():
        results_path.unlink()
    completed_keys = read_completed_keys(results_path)
    categories = set(args.category) if args.category else None
    pair_ids = read_pair_ids(root, args.split, args.layout_size, categories)
    pair_ids = sample_pair_ids(pair_ids, args.max_pairs, args.pair_sampling)
    total_pairs = len(pair_ids)
    pair_ids = shard_pair_ids(pair_ids, args.num_shards, args.shard_id)
    LOGGER.info(
        "Selected %d/%d image pairs for shard %d/%d",
        len(pair_ids),
        total_pairs,
        args.shard_id,
        args.num_shards,
    )

    predictor = None
    if not args.dry_run:
        predictor = LlavaOneVisionPredictor(
            model_path=args.model_path.resolve(),
            dtype_name=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        dry_run_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for pair_filename in pair_ids:
        annotation = load_annotation(root, args.split, pair_filename)
        image_dir = root / "JPEGImages" / annotation["category"]
        source_image = Image.open(image_dir / annotation["src_imname"]).convert("RGB")
        target_image = Image.open(image_dir / annotation["trg_imname"]).convert("RGB")
        source_input, source_scale = prepare_input_image(
            source_image, args.min_input_pixels
        )
        target_input, _ = prepare_input_image(target_image, args.min_input_pixels)

        for keypoint_index in selected_keypoints(annotation, args.keypoint_index):
            prediction_key = PredictionKey(pair_filename, keypoint_index)
            if prediction_key.value in completed_keys:
                continue

            source_point = annotation["src_kps"][keypoint_index]
            source_input_point = scale_point(source_point, source_scale)
            marked_source = draw_point_marker(
                source_input, source_input_point, args.marker_radius
            )
            prompt = build_prompt(
                annotation["category"],
                source_input_point,
                source_input.size,
                args.prompt_format,
            )
            if args.dry_run:
                stem = f"{pair_filename}_kp{keypoint_index}"
                marked_source.save(dry_run_dir / f"{stem}_source.png")
                target_input.save(dry_run_dir / f"{stem}_target.png")
                (dry_run_dir / f"{stem}_prompt.txt").write_text(prompt + "\n")
                processed += 1
                continue

            assert predictor is not None
            try:
                raw_output = predictor.predict(marked_source, target_input, prompt)
            except (RuntimeError, ValueError) as error:
                LOGGER.exception("Inference failed for %s", prediction_key.value)
                raw_output = f"INFERENCE_ERROR: {error}"

            prediction = prediction_from_output(
                annotation=annotation,
                split=args.split,
                keypoint_index=keypoint_index,
                target_size=target_image.size,
                source_input_size=source_input.size,
                target_input_size=target_input.size,
                raw_output=raw_output,
                alpha=args.pck_alpha,
            )
            with results_path.open("a") as results_file:
                results_file.write(json.dumps(asdict(prediction)) + "\n")
                results_file.flush()
            completed_keys.add(prediction_key.value)
            processed += 1

            if args.save_visualizations:
                category_dir = visualization_dir / annotation["category"]
                category_dir.mkdir(parents=True, exist_ok=True)
                visualization = draw_evaluation_visualization(
                    source_image,
                    target_image,
                    prediction,
                    args.marker_radius,
                )
                visualization.save(
                    category_dir / f"{pair_filename}_kp{keypoint_index}.jpg"
                )
            LOGGER.info(
                "%s output=%r error=%s correct=%s",
                prediction_key.value,
                raw_output,
                prediction.pixel_error,
                prediction.pck_correct,
            )

    if args.dry_run:
        LOGGER.info("Prepared %d dry-run queries in %s", processed, dry_run_dir)
    elif args.num_shards == 1:
        write_summary(results_path, summary_path, args)
    else:
        LOGGER.info(
            "Shard %d finished %d queries; summary will be built after merge",
            args.shard_id,
            processed,
        )


if __name__ == "__main__":
    main()
