#!/usr/bin/env python3
"""Summarize slime eval debug rollout files."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import struct
import zlib
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--debug-file", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-response-len", type=int, default=None)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--aggregate-dir", default=None)
    parser.add_argument("--combined-output-dir", default=None)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--sft-dir", default=None)
    parser.add_argument("--opd-dir", default=None)
    parser.add_argument("--sft-total-samples", type=int, default=24832)
    parser.add_argument("--sft-rollout-batch-size", type=int, default=256)
    parser.add_argument("--opd-rollout-batch-size", type=int, default=128)
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
    expected_samples: int | None,
) -> dict[str, Any]:
    import torch

    payload = torch.load(debug_file, map_location="cpu", weights_only=False)
    samples = payload.get("samples", [])
    if not samples:
        raise RuntimeError(f"No samples found in {debug_file}")
    if expected_samples is not None and len(samples) != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} samples in {debug_file}, found {len(samples)}")

    scores_and_parseable = [score_sample(sample) for sample in samples]
    rewards = [score for score, _ in scores_and_parseable]
    parseable_count = sum(1 for _, parseable in scores_and_parseable if parseable)
    response_lengths = [int(sample.get("response_length") or 0) for sample in samples]
    statuses = [str(sample.get("status", "")) for sample in samples]
    parse_failures = len(samples) - parseable_count

    cap_hits = 0
    for length, status in zip(response_lengths, statuses):
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


def read_stage_summaries(root: Path, phase: str) -> list[dict[str, Any]]:
    summaries = []
    if not root.exists():
        return summaries
    for path in sorted(root.glob("*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["phase"] = phase
        summaries.append(item)
    return summaries


def rollout_label(phase: str, train_samples: int | None, batch_size: int) -> str:
    if phase == "base" or train_samples is None:
        return "base"
    rollout_id = max(0, train_samples // batch_size - 1)
    return f"{phase.upper()} r{rollout_id}"


def combined_points(
    *,
    base_dir: Path | None,
    sft_dir: Path | None,
    opd_dir: Path | None,
    sft_total_samples: int,
    sft_rollout_batch_size: int,
    opd_rollout_batch_size: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if base_dir is not None:
        summaries.extend(read_stage_summaries(base_dir, "base"))
    if sft_dir is not None:
        summaries.extend(read_stage_summaries(sft_dir, "sft"))
    if opd_dir is not None:
        summaries.extend(read_stage_summaries(opd_dir, "opd"))

    points = []
    for item in summaries:
        phase = item["phase"]
        train_samples = item.get("train_samples")
        if phase == "base":
            x_value = 0
            sample_label = "0"
            step_label = "base"
        elif phase == "sft":
            x_value = int(train_samples or 0)
            sample_label = str(x_value)
            step_label = rollout_label("sft", x_value, sft_rollout_batch_size)
        else:
            opd_samples = int(train_samples or 0)
            x_value = sft_total_samples + opd_samples
            sample_label = f"SFT+{opd_samples}"
            step_label = rollout_label("opd", opd_samples, opd_rollout_batch_size)

        points.append(
            {
                "phase": phase,
                "stage": item.get("stage"),
                "train_samples": train_samples,
                "x_value": x_value,
                "sample_label": sample_label,
                "step_label": step_label,
                "accuracy": item.get("accuracy"),
                "accuracy_on_parseable": item.get("accuracy_on_parseable"),
                "parse_failure_rate": item.get("parse_failure_rate"),
                "cap_hit_rate": item.get("cap_hit_rate"),
                "avg_generated_tokens": item.get("avg_generated_tokens"),
            }
        )

    phase_order = {"base": 0, "sft": 1, "opd": 2}
    points.sort(key=lambda p: (p["x_value"], phase_order.get(str(p["phase"]), 9), str(p["stage"])))
    return points


def write_combined_csv(path: Path, points: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "stage",
                "train_samples",
                "x_value",
                "sample_label",
                "step_label",
                "accuracy",
                "accuracy_on_parseable",
                "avg_generated_tokens",
                "parse_failure_rate",
                "cap_hit_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(points)


def write_combined_svg(path: Path, points: list[dict[str, Any]], sft_total_samples: int) -> None:
    if not points:
        return

    width, height = 1120, 680
    margin_left, margin_right, margin_top, margin_bottom = 90, 45, 70, 160
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_x = max(max(int(point["x_value"]), sft_total_samples) for point in points)
    if max_x <= 0:
        max_x = 1

    def sx(x: int) -> float:
        return margin_left + (x / max_x) * plot_w

    def sy(y: float | None) -> float:
        y = 0.0 if y is None else max(0.0, min(1.0, float(y)))
        return margin_top + (1.0 - y) * plot_h

    sft_right = sx(min(sft_total_samples, max_x))
    bg_sft_w = max(0.0, sft_right - margin_left)
    bg_opd_x = sft_right
    bg_opd_w = max(0.0, margin_left + plot_w - bg_opd_x)

    by_phase = {"base": "#4b5563", "sft": "#2563eb", "opd": "#7c3aed"}
    mapped = [(sx(int(point["x_value"])), sy(point.get("accuracy")), point) for point in points]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in mapped)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{bg_sft_w:.1f}" height="{plot_h}" fill="#dbeafe" opacity="0.72"/>',
        f'<rect x="{bg_opd_x:.1f}" y="{margin_top}" width="{bg_opd_w:.1f}" height="{plot_h}" fill="#ede9fe" opacity="0.76"/>',
        f'<text x="{margin_left}" y="34" font-family="Arial, sans-serif" font-size="24" font-weight="700">MATH-500 Accuracy Curve</text>',
        f'<text x="{margin_left}" y="58" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">X-axis shows effective trained samples and rollout ids. Parse failures count wrong.</text>',
        f'<text x="{margin_left + 12}" y="{margin_top + 24}" font-family="Arial, sans-serif" font-size="13" fill="#1e40af">SFT segment</text>',
        f'<text x="{max(bg_opd_x + 12, margin_left + 120):.1f}" y="{margin_top + 24}" font-family="Arial, sans-serif" font-size="13" fill="#5b21b6">OPD segment</text>',
    ]

    for i in range(6):
        y_val = i / 5
        y = sy(y_val)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        lines.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">{y_val:.1f}</text>'
        )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111827" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#111827" stroke-width="1.5"/>'
    )
    if len(mapped) > 1:
        lines.append(f'<polyline points="{polyline}" fill="none" stroke="#111827" stroke-width="2.5"/>')

    for x, y, point in mapped:
        color = by_phase.get(str(point["phase"]), "#111827")
        label = html.escape(f'{point["step_label"]} / {point["sample_label"]}')
        stage = html.escape(str(point["stage"]))
        acc = point.get("accuracy")
        acc_label = "n/a" if acc is None else f"{float(acc) * 100:.1f}%"
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="white" stroke-width="2"/>')
        lines.append(f'<title>{stage}: accuracy {html.escape(acc_label)}, {label}</title>')
        tick_y = margin_top + plot_h
        lines.append(f'<line x1="{x:.1f}" y1="{tick_y}" x2="{x:.1f}" y2="{tick_y + 6}" stroke="#374151"/>')
        lines.append(
            f'<text transform="translate({x:.1f},{tick_y + 22}) rotate(-40)" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#111827">{label}</text>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{max(margin_top + 12, y - 12):.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{color}">{html.escape(acc_label)}</text>'
        )

    legend_x = width - 315
    legend_y = 30
    for idx, (label, color) in enumerate((("base", "#4b5563"), ("SFT", "#2563eb"), ("OPD", "#7c3aed"))):
        y = legend_y + idx * 22
        lines.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}"/>')
        lines.append(
            f'<text x="{legend_x + 14}" y="{y + 4}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{label}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_curve_png(path: Path, points: list[dict[str, Any]], sft_total_samples: int) -> None:
    if not points:
        return

    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 35, 35, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = bytearray([255] * width * height * 3)
    max_x = max(max(int(point["x_value"]), sft_total_samples) for point in points)
    if max_x <= 0:
        max_x = 1

    def fill_rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            for x in range(max(0, x0), min(width, x1)):
                put_px(image, width, height, x, y, color)

    def map_x(x: int) -> int:
        return margin_left + round((x / max_x) * plot_w)

    def map_y(y: float | None) -> int:
        y = 0.0 if y is None else max(0.0, min(1.0, float(y)))
        return margin_top + round((1.0 - y) * plot_h)

    sft_right = map_x(min(sft_total_samples, max_x))
    fill_rect(margin_left, margin_top, sft_right, margin_top + plot_h, (219, 234, 254))
    fill_rect(sft_right, margin_top, margin_left + plot_w, margin_top + plot_h, (237, 233, 254))

    axis = (32, 32, 32)
    grid = (210, 210, 210)
    line = (17, 24, 39)
    colors = {"base": (75, 85, 99), "sft": (37, 99, 235), "opd": (124, 58, 237)}

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

    mapped = [(map_x(int(point["x_value"])), map_y(point.get("accuracy")), point) for point in points]
    for a, b in zip(mapped, mapped[1:]):
        draw_line(image, width, height, (a[0], a[1]), (b[0], b[1]), line)
    for x, y, point in mapped:
        draw_dot(image, width, height, (x, y), 5, colors.get(str(point["phase"]), line))

    write_png(path, width, height, image)


def write_combined_report(
    out_dir: Path,
    *,
    base_dir: Path | None,
    sft_dir: Path | None,
    opd_dir: Path | None,
    sft_total_samples: int,
    sft_rollout_batch_size: int,
    opd_rollout_batch_size: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    points = combined_points(
        base_dir=base_dir,
        sft_dir=sft_dir,
        opd_dir=opd_dir,
        sft_total_samples=sft_total_samples,
        sft_rollout_batch_size=sft_rollout_batch_size,
        opd_rollout_batch_size=opd_rollout_batch_size,
    )
    if not points:
        raise RuntimeError("No per-stage summaries found for combined report")

    write_combined_csv(out_dir / "combined_accuracy_curve.csv", points)
    write_combined_svg(out_dir / "combined_accuracy_curve.svg", points, sft_total_samples)
    write_combined_curve_png(out_dir / "combined_accuracy_curve.png", points, sft_total_samples)
    return {
        "points": points,
        "artifacts": {
            "csv": str(out_dir / "combined_accuracy_curve.csv"),
            "svg": str(out_dir / "combined_accuracy_curve.svg"),
            "png": str(out_dir / "combined_accuracy_curve.png"),
        },
        "note": "Combined curve uses light blue shading for SFT and light purple shading for OPD.",
    }


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
    for a, b in zip(mapped, mapped[1:]):
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
    elif args.combined_output_dir:
        summary = write_combined_report(
            Path(args.combined_output_dir),
            base_dir=Path(args.base_dir) if args.base_dir else None,
            sft_dir=Path(args.sft_dir) if args.sft_dir else None,
            opd_dir=Path(args.opd_dir) if args.opd_dir else None,
            sft_total_samples=args.sft_total_samples,
            sft_rollout_batch_size=args.sft_rollout_batch_size,
            opd_rollout_batch_size=args.opd_rollout_batch_size,
        )
    else:
        if not args.stage or not args.debug_file:
            raise SystemExit("--stage and --debug-file are required unless --aggregate-dir or --combined-output-dir is used")
        summary = summarize_debug_file(
            args.stage,
            Path(args.debug_file),
            args.max_response_len,
            args.train_samples,
            args.expected_samples,
        )

    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
