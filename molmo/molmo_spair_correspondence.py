#!/usr/bin/env python3
"""Evaluate Molmo2-8B on SPair-71k semantic point correspondence.

This module reuses the dataset, metrics, visualization, resume, and progress
utilities from the neighboring LLaVA evaluator, while using Molmo2's official
multi-image chat API and native ``<points coords="...">`` output grammar.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from llava_spair_correspondence import (  # noqa: E402
    Prediction,
    PredictionKey,
    coordinate_to_pixel,
    count_selected_keypoints,
    draw_pair_evaluation_visualization,
    draw_point_marker,
    load_annotation,
    point_error,
    pck_threshold,
    prepare_input_image,
    read_completed_keys,
    read_pair_ids,
    read_prediction_records,
    sample_pair_ids,
    scale_point,
    selected_keypoints,
    shard_pair_ids,
    write_progress,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

LOGGER = logging.getLogger("molmo_spair_correspondence")
COORDINATE_SCALE = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Molmo2-8B correspondence evaluation on SPair-71k."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/data/shared-vilab/pretrained_models/VLM_models/Molmo2-8B"
        ),
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
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument(
        "--pair-sampling",
        choices=("first", "stratified"),
        default="stratified",
    )
    parser.add_argument("--category", action="append")
    parser.add_argument("--keypoint-index", type=int)
    parser.add_argument("--pck-alpha", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--min-input-pixels",
        type=int,
        default=0,
        help="Optional pre-upscaling area; 0 lets Molmo2 preprocess originals.",
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--marker-radius", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "spair_correspondence_results",
    )
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.max_pairs < 0:
        parser.error("--max-pairs must be >= 0")
    if args.pck_alpha <= 0:
        parser.error("--pck-alpha must be > 0")
    if args.min_input_pixels < 0:
        parser.error("--min-input-pixels must be >= 0")
    if args.num_shards <= 0:
        parser.error("--num-shards must be > 0")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must satisfy 0 <= shard-id < num-shards")
    return args


def build_prompt(category: str) -> str:
    """Ask for one target point using Molmo2's native multi-image grammar."""
    return (
        "You are solving semantic point correspondence between two instances "
        f"of the same object category ({category}). Image 1 is the SOURCE and "
        "Image 2 is the TARGET. The red cross in Image 1 marks the query point. "
        "Find the exact semantically corresponding anatomical or structural "
        "point in Image 2, accounting for viewpoint, articulation, scale, and "
        "deformation. Do not match the red color itself. Point only in Image 2. "
        "Return exactly one point using Molmo2 multi-image point format, with "
        "image id 2, point id 0, and integer coordinates on the 0-1000 scale. "
        'Answer only: <points coords="2 0 <x> <y>">corresponding point</points>'
    )


def parse_target_point(text: str) -> tuple[float, float] | None:
    """Extract point-id 0 in image 2 from Molmo2 native point/track output."""
    coord_matches = re.findall(
        r"<(?:points|tracks)\b[^>]*\bcoords\s*=\s*[\"']([^\"']+)[\"'][^>]*>",
        text,
        flags=re.IGNORECASE,
    )
    candidates: list[tuple[float, float]] = []
    for coords in coord_matches:
        for image_group in re.split(r"[\t:;,]", coords):
            fields = image_group.strip().split()
            if len(fields) < 4:
                continue
            try:
                image_id = int(float(fields[0]))
            except ValueError:
                continue
            if image_id != 2:
                continue
            point_fields = fields[1:]
            for index in range(0, len(point_fields) - 2, 3):
                try:
                    point_id = int(float(point_fields[index]))
                    x = float(point_fields[index + 1])
                    y = float(point_fields[index + 2])
                except ValueError:
                    continue
                if point_id == 0:
                    candidates.append((x, y))
    if candidates:
        return candidates[-1]

    # Tolerate a bare coordinate pair if the model ignores the requested XML.
    pairs = re.findall(
        r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*[,;]\s*"
        r"(-?\d+(?:\.\d+)?)\s*[\)\]]",
        text,
    )
    if pairs:
        return float(pairs[-1][0]), float(pairs[-1][1])
    return None


class Molmo2Predictor:
    """Thin wrapper around the official Transformers multi-image API."""

    def __init__(
        self,
        model_path: Path,
        dtype_name: str,
        device_map: str,
        attn_implementation: str,
        max_new_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        dtype: Any = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
        LOGGER.info("Loading Molmo2 processor from %s", model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        LOGGER.info("Loading Molmo2 model from %s", model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        ).eval()

    def _input_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device(
                "cuda" if self.torch.cuda.is_available() else "cpu"
            )

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
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": source_image},
                    {"type": "image", "image": target_image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
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
        generated_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


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
    normalized_point = parse_target_point(raw_output)
    predicted_point = None
    parse_error = None
    if normalized_point is not None:
        predicted_point = coordinate_to_pixel(*normalized_point, target_size)
        if predicted_point is None:
            parse_error = "Predicted coordinates are outside [0, 1000]"
    else:
        parse_error = "Could not parse image-2 point from Molmo2 output"

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
            list(map(float, predicted_point)) if predicted_point is not None else None
        ),
        source_input_size=list(source_input_size),
        target_input_size=list(target_input_size),
        target_visible=True if predicted_point is not None else None,
        valid_prediction=predicted_point is not None,
        pixel_error=error,
        pck_threshold=threshold,
        pck_correct=error is not None and error <= threshold,
        raw_output=raw_output,
        parse_error=parse_error,
    )


def write_summary(
    results_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
) -> None:
    records: list[dict[str, Any]] = []
    if results_path.is_file():
        for line in results_path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    valid = [item for item in records if item.get("valid_prediction")]
    correct = [item for item in records if item.get("pck_correct")]
    errors = [
        float(item["pixel_error"])
        for item in valid
        if item.get("pixel_error") is not None
    ]
    by_category: dict[str, dict[str, float | int]] = {}
    for category in sorted({str(item["category"]) for item in records}):
        category_records = [
            item for item in records if item.get("category") == category
        ]
        category_valid = [
            item for item in category_records if item.get("valid_prediction")
        ]
        category_correct = [
            item for item in category_records if item.get("pck_correct")
        ]
        by_category[category] = {
            "num_predictions": len(category_records),
            "num_valid_predictions": len(category_valid),
            "parse_rate": (
                len(category_valid) / len(category_records)
                if category_records
                else 0.0
            ),
            "accuracy_pck": (
                len(category_correct) / len(category_records)
                if category_records
                else 0.0
            ),
        }
    summary = {
        "model_family": "Molmo2",
        "model_path": str(args.model_path),
        "split": args.split,
        "layout_size": args.layout_size,
        "pair_sampling": args.pair_sampling,
        "pck_alpha": args.pck_alpha,
        "accuracy_metric": (
            "PCK: pixel error <= alpha * max(target bounding-box width, height)"
        ),
        "prompt_format": "molmo2-native-multi-image-points",
        "min_input_pixels": args.min_input_pixels,
        "num_predictions": len(records),
        "num_valid_predictions": len(valid),
        "parse_rate": len(valid) / len(records) if records else 0.0,
        "accuracy_pck": len(correct) / len(records) if records else 0.0,
        "pck_among_valid": len(correct) / len(valid) if valid else 0.0,
        "mean_pixel_error_valid": sum(errors) / len(errors) if errors else None,
        "by_category": by_category,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Summary: %s", json.dumps(summary))


def _save_pair_visualization(
    output_dir: Path,
    split: str,
    annotation: dict[str, Any],
    source_image: Image.Image,
    target_image: Image.Image,
    predictions: list[Prediction],
    marker_radius: int,
) -> None:
    category_dir = output_dir / "visualizations" / split / annotation["category"]
    category_dir.mkdir(parents=True, exist_ok=True)
    visualization = draw_pair_evaluation_visualization(
        source_image,
        target_image,
        sorted(predictions, key=lambda item: item.keypoint_index),
        marker_radius,
    )
    visualization.save(category_dir / f"{annotation['filename']}.jpg")


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

    if args.summary_only:
        write_summary(output_dir / f"{args.split}_predictions.jsonl", summary_path, args)
        return
    if args.overwrite and results_path.exists():
        results_path.unlink()

    completed_keys = read_completed_keys(results_path)
    existing_predictions = read_prediction_records(results_path)
    categories = set(args.category) if args.category else None
    pair_ids = sample_pair_ids(
        read_pair_ids(root, args.split, args.layout_size, categories),
        args.max_pairs,
        args.pair_sampling,
    )
    total_pairs = len(pair_ids)
    pair_ids = shard_pair_ids(pair_ids, args.num_shards, args.shard_id)
    LOGGER.info(
        "Selected %d/%d pairs for shard %d/%d",
        len(pair_ids),
        total_pairs,
        args.shard_id,
        args.num_shards,
    )
    annotations = {
        pair_id: load_annotation(root, args.split, pair_id) for pair_id in pair_ids
    }
    total_queries = sum(
        count_selected_keypoints(annotation, args.keypoint_index)
        for annotation in annotations.values()
    )
    done_queries = sum(
        PredictionKey(pair_id, kp_index).value in completed_keys
        for pair_id, annotation in annotations.items()
        for kp_index in selected_keypoints(annotation, args.keypoint_index)
    )
    progress_path = output_dir / "progress" / f"shard{args.shard_id:02d}.json"
    write_progress(
        progress_path,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        total_pairs=len(pair_ids),
        done_pairs=0,
        total_queries=total_queries,
        done_queries=done_queries,
    )

    predictor = None
    if args.dry_run:
        (output_dir / "dry_run").mkdir(parents=True, exist_ok=True)
    else:
        predictor = Molmo2Predictor(
            args.model_path.resolve(),
            args.dtype,
            args.device_map,
            args.attn_implementation,
            args.max_new_tokens,
        )

    pair_iter: Iterable[str] = pair_ids
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            pair_ids,
            desc=f"molmo-shard{args.shard_id:02d}",
            unit="pair",
        )
        pair_iter = progress_bar

    for done_pairs, pair_id in enumerate(pair_iter, start=1):
        annotation = annotations[pair_id]
        image_dir = root / "JPEGImages" / annotation["category"]
        source_image = Image.open(image_dir / annotation["src_imname"]).convert("RGB")
        target_image = Image.open(image_dir / annotation["trg_imname"]).convert("RGB")
        source_input, source_scale = prepare_input_image(
            source_image, args.min_input_pixels
        )
        target_input, _ = prepare_input_image(target_image, args.min_input_pixels)
        pair_predictions: list[Prediction] = []

        for kp_index in selected_keypoints(annotation, args.keypoint_index):
            key = PredictionKey(pair_id, kp_index)
            if key.value in completed_keys:
                previous = existing_predictions.get(key.value)
                if previous is not None:
                    pair_predictions.append(previous)
                continue

            source_point = scale_point(annotation["src_kps"][kp_index], source_scale)
            marked_source = draw_point_marker(
                source_input, source_point, args.marker_radius
            )
            prompt = build_prompt(annotation["category"])
            if args.dry_run:
                stem = f"{pair_id}_kp{kp_index}"
                marked_source.save(output_dir / "dry_run" / f"{stem}_source.png")
                target_input.save(output_dir / "dry_run" / f"{stem}_target.png")
                (output_dir / "dry_run" / f"{stem}_prompt.txt").write_text(
                    prompt + "\n"
                )
                done_queries += 1
                continue

            assert predictor is not None
            try:
                raw_output = predictor.predict(marked_source, target_input, prompt)
            except (RuntimeError, ValueError) as error:
                LOGGER.exception("Inference failed for %s", key.value)
                raw_output = f"INFERENCE_ERROR: {error}"
            prediction = prediction_from_output(
                annotation,
                args.split,
                kp_index,
                target_image.size,
                source_input.size,
                target_input.size,
                raw_output,
                args.pck_alpha,
            )
            with results_path.open("a") as results_file:
                results_file.write(json.dumps(asdict(prediction)) + "\n")
                results_file.flush()
            completed_keys.add(key.value)
            existing_predictions[key.value] = prediction
            pair_predictions.append(prediction)
            done_queries += 1
            LOGGER.info(
                "%s output=%r error=%s correct=%s",
                key.value,
                raw_output,
                prediction.pixel_error,
                prediction.pck_correct,
            )

        if args.save_visualizations and pair_predictions and not args.dry_run:
            _save_pair_visualization(
                output_dir,
                args.split,
                annotation,
                source_image,
                target_image,
                pair_predictions,
                args.marker_radius,
            )
        write_progress(
            progress_path,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            total_pairs=len(pair_ids),
            done_pairs=done_pairs,
            total_queries=total_queries,
            done_queries=done_queries,
            current_pair=pair_id,
        )
        if progress_bar is not None:
            progress_bar.set_postfix(
                queries=f"{done_queries}/{total_queries}", refresh=False
            )

    if progress_bar is not None:
        progress_bar.close()
    write_progress(
        progress_path,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        total_pairs=len(pair_ids),
        done_pairs=len(pair_ids),
        total_queries=total_queries,
        done_queries=done_queries,
    )
    if not args.dry_run and args.num_shards == 1:
        write_summary(results_path, summary_path, args)


if __name__ == "__main__":
    main()
