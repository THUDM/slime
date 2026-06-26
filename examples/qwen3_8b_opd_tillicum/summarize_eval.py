#!/usr/bin/env python3
"""Summarize slime eval debug rollout files."""

from __future__ import annotations

import argparse
import json
import math
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
    return parser.parse_args()


def reward_value(sample: dict[str, Any]) -> float:
    reward = sample.get("reward", 0.0)
    if isinstance(reward, dict):
        for key in ("reward", "score", "acc", "accuracy"):
            if key in reward:
                return float(reward[key])
        return 0.0
    return float(reward or 0.0)


def extract_answer_or_none(response: str) -> str | None:
    try:
        from slime.rollout.rm_hub.math_utils import extract_answer
    except Exception:
        return None

    if "</think>" in response:
        response = response.split("</think>")[-1]
    elif "###Response" in response:
        response = response.split("###Response", 1)[1]
    return extract_answer(response)


def summarize_debug_file(stage: str, debug_file: Path, max_response_len: int | None) -> dict[str, Any]:
    payload = torch.load(debug_file, map_location="cpu", weights_only=False)
    samples = payload.get("samples", [])
    if not samples:
        raise RuntimeError(f"No samples found in {debug_file}")

    rewards = [reward_value(sample) for sample in samples]
    response_lengths = [int(sample.get("response_length") or 0) for sample in samples]
    statuses = [str(sample.get("status", "")) for sample in samples]

    parse_failures = 0
    for sample in samples:
        response = str(sample.get("response", ""))
        if extract_answer_or_none(response) is None:
            parse_failures += 1

    cap_hits = 0
    for length, status in zip(response_lengths, statuses, strict=True):
        if status == "truncated" or (max_response_len is not None and length >= max_response_len):
            cap_hits += 1

    n = len(samples)
    accuracy = sum(rewards) / n
    return {
        "stage": stage,
        "debug_file": str(debug_file),
        "n": n,
        "accuracy": accuracy,
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

    return {
        "num_repeats_per_stage": 1,
        "stages": summaries,
        "note": "Eval repeat count is 1, so per-stage std_accuracy is N/A.",
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.aggregate_dir:
        summary = aggregate(Path(args.aggregate_dir))
    else:
        if not args.stage or not args.debug_file:
            raise SystemExit("--stage and --debug-file are required unless --aggregate-dir is used")
        summary = summarize_debug_file(args.stage, Path(args.debug_file), args.max_response_len)

    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
