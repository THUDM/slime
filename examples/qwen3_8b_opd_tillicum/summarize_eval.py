#!/usr/bin/env python3
"""Summarize slime eval debug rollout files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import zlib
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--debug-file", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-response-len", type=int, default=None)
    parser.add_argument("--aggregate-dir", default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    return parser.parse_args()


FINAL_ANSWER_RE = re.compile(r"(?i)\b(?:final\s+answer|answer)\s*(?:is|:)\s*(?P<answer>.+)$")


def answer_segment(response: str) -> str:
    if "</think>" in response:
        return response.rsplit("</think>", 1)[1]
    if "###Response" in response:
        return response.split("###Response", 1)[1]
    return response


def clean_final_answer_candidate(candidate: str) -> str:
    candidate = candidate.strip()
    candidate = re.split(r"<\|endoftext\|>|</s>|<pad>", candidate, maxsplit=1)[0].strip()
    candidate = candidate.split(". ", 1)[0].strip()
    candidate = candidate.split("\n", 1)[0].strip()

    for left, right in (("\\(", "\\)"), ("\\[", "\\]"), ("$", "$")):
        if candidate.startswith(left) and candidate.endswith(right):
            candidate = candidate[len(left) : -len(right)].strip()

    while candidate and candidate[-1] in ".,;:":
        candidate = candidate[:-1].strip()
    return candidate


def extract_prediction(response: str) -> str | None:
    from slime.rollout.rm_hub.math_utils import extract_answer

    segment = answer_segment(response)
    boxed = extract_answer(segment)
    if boxed is not None:
        return boxed

    for line in reversed(segment.splitlines()):
        match = FINAL_ANSWER_RE.search(line.strip())
        if not match:
            continue
        candidate = clean_final_answer_candidate(match.group("answer"))
        if candidate and len(candidate) <= 120:
            return candidate
    return None


def extract_ground_truth(label: Any) -> str:
    from slime.rollout.rm_hub.math_utils import extract_answer

    ground_truth = str(label)
    if "\\boxed" in ground_truth:
        boxed = extract_answer(ground_truth)
        if boxed is not None:
            return boxed
    return ground_truth


def score_sample(sample: dict[str, Any]) -> tuple[float, bool]:
    from slime.rollout.rm_hub.math_utils import grade_answer_mathd, grade_answer_sympy

    response = str(sample.get("response", ""))
    label = sample.get("label", "")
    prediction = extract_prediction(response)
    if prediction is None:
        return 0.0, False

    ground_truth = extract_ground_truth(label)
    is_correct = grade_answer_mathd(prediction, ground_truth) or grade_answer_sympy(prediction, ground_truth)
    return float(is_correct), True


def summarize_debug_file(
    stage: str,
    debug_file: Path,
    max_response_len: int | None,
    train_samples: int | None,
) -> dict[str, Any]:
    payload = torch.load(debug_file, map_location="cpu", weights_only=False)
    samples = payload.get("samples", [])
    if not samples:
        raise RuntimeError(f"No samples found in {debug_file}")

    scores_and_parseable = [score_sample(sample) for sample in samples]
    rewards = [score for score, _ in scores_and_parseable]
    parseable_count = sum(1 for _, parseable in scores_and_parseable if parseable)
    response_lengths = [int(sample.get("response_length") or 0) for sample in samples]
    statuses = [str(sample.get("status", "")) for sample in samples]
    parse_failures = len(samples) - parseable_count

    cap_hits = 0
    for length, status in zip(response_lengths, statuses, strict=True):
        if status == "truncated" or (max_response_len is not None and length >= max_response_len):
            cap_hits += 1

    n = len(samples)
    accuracy = sum(rewards) / n
    accuracy_on_parseable = sum(rewards) / parseable_count if parseable_count else None
    return {
        "stage": stage,
        "train_samples": train_samples,
        "debug_file": str(debug_file),
        "n": n,
        "correct_count": int(sum(rewards)),
        "parseable_count": parseable_count,
        "accuracy": accuracy,
        "accuracy_on_parseable": accuracy_on_parseable,
        "mean_accuracy": accuracy,
        "std_accuracy": None,
        "avg_generated_tokens": sum(response_lengths) / n,
        "max_generated_tokens": max(response_lengths),
        "parse_failure_rate": parse_failures / n,
        "cap_hit_rate": cap_hits / n,
    }


def aggregate(aggregate_dir: Path) -> dict[str, Any]:
    summaries = []
    for path in sorted(aggregate_dir.glob("*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        raise RuntimeError(f"No per-stage summaries found under {aggregate_dir}")

    summaries.sort(key=lambda item: (item.get("train_samples") is None, item.get("train_samples") or 0, item["stage"]))
    write_accuracy_curve_csv(aggregate_dir / "accuracy_curve.csv", summaries)
    write_accuracy_curve_png(aggregate_dir / "accuracy_curve.png", summaries)

    return {
        "num_repeats_per_stage": 1,
        "stages": summaries,
        "note": "Eval repeat count is 1, so per-stage std_accuracy is N/A.",
    }


def write_accuracy_curve_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "train_samples",
                "accuracy",
                "accuracy_on_parseable",
                "avg_generated_tokens",
                "parse_failure_rate",
                "cap_hit_rate",
            ],
        )
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "stage": item.get("stage"),
                    "train_samples": item.get("train_samples"),
                    "accuracy": item.get("accuracy"),
                    "accuracy_on_parseable": item.get("accuracy_on_parseable"),
                    "avg_generated_tokens": item.get("avg_generated_tokens"),
                    "parse_failure_rate": item.get("parse_failure_rate"),
                    "cap_hit_rate": item.get("cap_hit_rate"),
                }
            )


def put_px(image: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        idx = (y * width + x) * 3
        image[idx : idx + 3] = bytes(color)


def draw_line(
    image: bytearray,
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        put_px(image, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_dot(
    image: bytearray,
    width: int,
    height: int,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    cx, cy = center
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                put_px(image, width, height, x, y, color)


def write_png(path: Path, width: int, height: int, image: bytearray) -> None:
    rows = []
    row_bytes = width * 3
    for y in range(height):
        rows.append(b"\x00" + bytes(image[y * row_bytes : (y + 1) * row_bytes]))
    raw = b"".join(rows)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk("IHDR".encode(), struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk("IDAT".encode(), zlib.compress(raw, level=9))
    png += chunk("IEND".encode(), b"")
    path.write_bytes(png)


def write_accuracy_curve_png(path: Path, summaries: list[dict[str, Any]]) -> None:
    points = [
        (item.get("train_samples"), float(item.get("accuracy", 0.0)))
        for item in summaries
        if item.get("train_samples") is not None
    ]
    if not points:
        return

    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 35, 35, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = bytearray([255] * width * height * 3)

    axis = (32, 32, 32)
    grid = (225, 225, 225)
    line = (34, 102, 190)
    dot = (190, 60, 60)

    x_min = min(x for x, _ in points)
    x_max = max(x for x, _ in points)
    if x_min == x_max:
        x_min = 0
    y_min, y_max = 0.0, 1.0

    def map_x(x: int) -> int:
        if x_max == x_min:
            return margin_left + plot_w // 2
        return margin_left + round((x - x_min) / (x_max - x_min) * plot_w)

    def map_y(y: float) -> int:
        return margin_top + round((y_max - max(y_min, min(y_max, y))) / (y_max - y_min) * plot_h)

    for i in range(6):
        y = margin_top + round(i / 5 * plot_h)
        draw_line(image, width, height, (margin_left, y), (width - margin_right, y), grid)
    draw_line(image, width, height, (margin_left, margin_top), (margin_left, height - margin_bottom), axis)
    draw_line(
        image,
        width,
        height,
        (margin_left, height - margin_bottom),
        (width - margin_right, height - margin_bottom),
        axis,
    )

    mapped = [(map_x(x), map_y(y)) for x, y in points]
    for a, b in zip(mapped, mapped[1:], strict=False):
        draw_line(image, width, height, a, b, line)
    for point in mapped:
        draw_dot(image, width, height, point, 5, dot)

    write_png(path, width, height, image)


def main() -> None:
    args = parse_args()
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.aggregate_dir:
        summary = aggregate(Path(args.aggregate_dir))
    else:
        if not args.stage or not args.debug_file:
            raise SystemExit("--stage and --debug-file are required unless --aggregate-dir is used")
        summary = summarize_debug_file(args.stage, Path(args.debug_file), args.max_response_len, args.train_samples)

    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
