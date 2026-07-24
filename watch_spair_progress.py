#!/usr/bin/env python3
"""Aggregate multi-GPU SPair shard progress into one live bar."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def render_bar(fraction: float, width: int = 40) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(width * fraction))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def load_shard_progress(progress_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(progress_dir.glob("shard*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open() as handle:
        for _ in handle:
            count += 1
    return count


def fallback_from_predictions(output_dir: Path) -> list[dict]:
    """Estimate progress from shard prediction files when progress/*.json is absent."""
    shards_dir = output_dir / "shards"
    records: list[dict] = []
    if shards_dir.is_dir():
        for shard_dir in sorted(shards_dir.glob("shard*")):
            try:
                shard_id = int(shard_dir.name.replace("shard", ""))
            except ValueError:
                continue
            pred = shard_dir / "test_predictions.jsonl"
            done = count_jsonl_lines(pred)
            records.append(
                {
                    "shard_id": shard_id,
                    "total_pairs": 0,
                    "done_pairs": 0,
                    "total_queries": 0,
                    "done_queries": done,
                    "current_pair": f"(from {pred.name}: {done} lines)",
                }
            )
    else:
        pred = output_dir / "test_predictions.jsonl"
        done = count_jsonl_lines(pred)
        if done:
            records.append(
                {
                    "shard_id": 0,
                    "total_pairs": 0,
                    "done_pairs": 0,
                    "total_queries": 0,
                    "done_queries": done,
                    "current_pair": f"(from {pred.name}: {done} lines)",
                }
            )
    return records


def format_status(records: list[dict], *, fallback: bool = False) -> str:
    if not records:
        return "Waiting for shard progress files..."

    total_pairs = sum(int(item.get("total_pairs", 0)) for item in records)
    done_pairs = sum(int(item.get("done_pairs", 0)) for item in records)
    total_queries = sum(int(item.get("total_queries", 0)) for item in records)
    done_queries = sum(int(item.get("done_queries", 0)) for item in records)
    pair_frac = done_pairs / total_pairs if total_pairs else 0.0
    query_frac = done_queries / total_queries if total_queries else 0.0

    lines: list[str] = []
    if fallback:
        lines.append(
            "(fallback mode: counting prediction jsonl lines; "
            "restart run for full progress bars)"
        )
        lines.append(f"Completed queries so far: {done_queries}")
        lines.append("")
    else:
        lines.extend(
            [
                f"Overall pairs   {render_bar(pair_frac)} "
                f"{done_pairs}/{total_pairs} ({100.0 * pair_frac:5.1f}%)",
                f"Overall queries {render_bar(query_frac)} "
                f"{done_queries}/{total_queries} ({100.0 * query_frac:5.1f}%)",
                "",
            ]
        )
    for item in records:
        shard_id = int(item.get("shard_id", -1))
        tp = int(item.get("total_pairs", 0))
        dp = int(item.get("done_pairs", 0))
        tq = int(item.get("total_queries", 0))
        dq = int(item.get("done_queries", 0))
        frac = dp / tp if tp else (dq / tq if tq else 0.0)
        current = item.get("current_pair") or "-"
        if fallback:
            lines.append(f"  shard{shard_id:02d}  queries_done={dq}  {current}")
        else:
            lines.append(
                f"  shard{shard_id:02d} {render_bar(frac, width=24)} "
                f"pairs {dp}/{tp}  queries {dq}/{tq}  current={current}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("spair_correspondence_results"),
        help="SPair output directory containing progress/",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print once and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    progress_dir = output_dir / "progress"
    previous = ""
    while True:
        records = load_shard_progress(progress_dir)
        fallback = False
        if not records:
            records = fallback_from_predictions(output_dir)
            fallback = bool(records)
        status = format_status(records, fallback=fallback)
        if status != previous:
            if not args.once and previous:
                # Clear previous multi-line block roughly.
                sys.stdout.write("\033[2J\033[H")
            print(status, flush=True)
            previous = status
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
