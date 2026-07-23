#!/usr/bin/env python3
"""Aggregate how often predicted point tracks move across experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("davis_tracking_multi_sequence"),
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=1.0,
        help=(
            "A normalized trajectory span above this value is classified as "
            "moving. Coordinates use the model's 0-1000 scale."
        ),
    )
    parser.add_argument(
        "--meaningful-threshold",
        type=float,
        default=10.0,
        help=(
            "Second, stricter movement threshold on the 0-1000 coordinate "
            "scale. Default 10 corresponds to roughly 1%% of an axis."
        ),
    )
    args = parser.parse_args()
    if args.movement_threshold < 0 or args.meaningful_threshold < 0:
        parser.error("movement thresholds must be >= 0")
    return args


def trajectory_statistics(
    parsed_tracks: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    point_histories: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for timestamp in sorted(parsed_tracks, key=float):
        frame_points = parsed_tracks[timestamp]
        for point_id, coordinates in frame_points.items():
            if len(coordinates) < 2:
                continue
            point_histories[str(point_id)].append(
                (float(coordinates[0]), float(coordinates[1]))
            )

    per_point = {}
    all_spans = []
    for point_id, coordinates in sorted(point_histories.items()):
        xs = [point[0] for point in coordinates]
        ys = [point[1] for point in coordinates]
        span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        unique_coordinates = len(set(coordinates))
        per_point[point_id] = {
            "num_predictions": len(coordinates),
            "num_unique_coordinates": unique_coordinates,
            "span_normalized": span,
            "first_coordinate": list(coordinates[0]),
            "last_coordinate": list(coordinates[-1]),
        }
        all_spans.append(span)

    return {
        "num_timestamps": len(parsed_tracks),
        "num_point_ids": len(point_histories),
        "max_span_normalized": max(all_spans) if all_spans else None,
        "per_point": per_point,
    }


def identify_case(
    results_root: Path,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, str]:
    relative_parts = config_path.relative_to(results_root).parts
    sequence = relative_parts[0] if relative_parts else "unknown"
    experiment_set = (
        relative_parts[1] if len(relative_parts) > 1 else "unknown"
    )
    if experiment_set == "experiment1":
        case_number = str(config.get("number", relative_parts[-2]))
        case_key = f"case_{case_number}_{config.get('name', relative_parts[-2])}"
        group = "experiment1"
        folder = relative_parts[-2]
    else:
        group = str(config.get("group", relative_parts[-3]))
        folder = str(config.get("folder", relative_parts[-2]))
        case_key = f"{group}/{folder}"
    return {
        "sequence": sequence,
        "experiment_set": experiment_set,
        "group": group,
        "folder": folder,
        "case_key": case_key,
    }


def analyze_case(
    results_root: Path,
    config_path: Path,
    movement_threshold: float,
    meaningful_threshold: float,
) -> dict[str, Any]:
    case_dir = config_path.parent
    config = json.loads(config_path.read_text())
    identity = identify_case(results_root, config_path, config)
    tracks_path = case_dir / "tracks.json"

    if (case_dir / "skipped.txt").exists():
        status = "skipped"
    elif (case_dir / "error.txt").exists():
        status = "failed"
    elif not tracks_path.exists():
        status = "not_tracking_output"
    else:
        status = "completed"

    row: dict[str, Any] = {
        **identity,
        "name": config.get("name"),
        "input_kind": config.get("input_kind"),
        "num_frames": config.get("num_frames"),
        "max_pixels": config.get("max_pixels"),
        "max_new_tokens": config.get("max_new_tokens"),
        "prompt": config.get("prompt"),
        "case_dir": str(case_dir),
        "status": status,
        "num_timestamps": 0,
        "num_point_ids": 0,
        "max_span_normalized": None,
        "coordinates_changed": None,
        "moving": None,
        "meaningfully_moving": None,
        "per_point": {},
    }
    if status != "completed":
        return row

    payload = json.loads(tracks_path.read_text())
    statistics = trajectory_statistics(payload.get("parsed_tracks", {}))
    span = statistics["max_span_normalized"]
    valid_track = (
        statistics["num_timestamps"] >= 2
        and statistics["num_point_ids"] >= 1
        and span is not None
    )
    row.update(statistics)
    row["status"] = "valid_track" if valid_track else "invalid_track"
    if valid_track:
        row["coordinates_changed"] = bool(span > 0)
        row["moving"] = bool(span > movement_threshold)
        row["meaningfully_moving"] = bool(span > meaningful_threshold)
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["experiment_set"], row["case_key"])].append(row)

    aggregates = []
    for (experiment_set, case_key), case_rows in sorted(grouped.items()):
        valid_rows = [row for row in case_rows if row["status"] == "valid_track"]
        changed_rows = [
            row for row in valid_rows if row["coordinates_changed"]
        ]
        moving_rows = [row for row in valid_rows if row["moving"]]
        meaningful_rows = [
            row for row in valid_rows if row["meaningfully_moving"]
        ]
        spans = [
            float(row["max_span_normalized"])
            for row in valid_rows
            if row["max_span_normalized"] is not None
        ]
        aggregates.append(
            {
                "experiment_set": experiment_set,
                "case_key": case_key,
                "name": case_rows[0]["name"],
                "num_sequences": len(case_rows),
                "num_valid_tracks": len(valid_rows),
                "num_failed_or_invalid": len(case_rows) - len(valid_rows),
                "num_coordinates_changed": len(changed_rows),
                "coordinate_change_rate": (
                    len(changed_rows) / len(valid_rows) if valid_rows else None
                ),
                "num_moving": len(moving_rows),
                "movement_rate": (
                    len(moving_rows) / len(valid_rows) if valid_rows else None
                ),
                "num_meaningfully_moving": len(meaningful_rows),
                "meaningful_movement_rate": (
                    len(meaningful_rows) / len(valid_rows)
                    if valid_rows
                    else None
                ),
                "mean_span_normalized": (
                    sum(spans) / len(spans) if spans else None
                ),
                "max_span_normalized": max(spans) if spans else None,
                "moving_sequences": [
                    row["sequence"] for row in moving_rows
                ],
                "static_sequences": [
                    row["sequence"]
                    for row in valid_rows
                    if not row["moving"]
                ],
            }
        )
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_keys = [
        "sequence",
        "experiment_set",
        "group",
        "folder",
        "case_key",
        "name",
        "input_kind",
        "num_frames",
        "max_pixels",
        "max_new_tokens",
        "status",
        "num_timestamps",
        "num_point_ids",
        "max_span_normalized",
        "coordinates_changed",
        "moving",
        "meaningfully_moving",
        "case_dir",
    ]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_keys = [
        "experiment_set",
        "case_key",
        "name",
        "num_sequences",
        "num_valid_tracks",
        "num_failed_or_invalid",
        "num_coordinates_changed",
        "coordinate_change_rate",
        "num_moving",
        "movement_rate",
        "num_meaningfully_moving",
        "meaningful_movement_rate",
        "mean_span_normalized",
        "max_span_normalized",
    ]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    config_paths = sorted(results_root.glob("*/experiment*/**/config.json"))
    rows = [
        analyze_case(
            results_root,
            config_path,
            args.movement_threshold,
            args.meaningful_threshold,
        )
        for config_path in config_paths
    ]
    aggregates = aggregate_rows(rows)
    payload = {
        "results_root": str(results_root),
        "definition": {
            "coordinate_scale": [0, 1000],
            "movement_threshold": args.movement_threshold,
            "meaningful_threshold": args.meaningful_threshold,
            "movement_rate_denominator": (
                "runs with >=2 parsed timestamps and >=1 point ID"
            ),
            "note": (
                "Movement rate measures whether predicted coordinates change; "
                "it does not measure tracking correctness."
            ),
        },
        "num_cases_found": len(rows),
        "by_experiment": aggregates,
        "per_sequence_case": rows,
    }
    (results_root / "movement_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_csv(results_root / "movement_analysis.csv", rows)
    write_aggregate_csv(
        results_root / "movement_analysis_by_experiment.csv",
        aggregates,
    )

    print(f"Analyzed {len(rows)} runs under {results_root}")
    print(f"Saved: {results_root / 'movement_analysis.json'}")
    print(f"Saved: {results_root / 'movement_analysis.csv'}")
    print(
        "Saved: "
        f"{results_root / 'movement_analysis_by_experiment.csv'}"
    )


if __name__ == "__main__":
    main()
