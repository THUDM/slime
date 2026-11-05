#!/usr/bin/env python3
"""Build deterministic ``--load-debug-rollout-data`` files for slime.

The input is JSONL with one token sequence per line. A sequence is stored as
one :class:`slime.utils.types.Sample` with the first token treated as the
prompt boundary and the remaining tokens as supervised targets. This is the
same ``seq[:-1]``/``seq[1:]`` convention used by the Weaver benchmark.

The generator deliberately rejects duplicate sequences and refuses to recycle
rows. That makes a benchmark fail early instead of silently measuring the
same example repeatedly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from slime.utils.types import Sample


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSONL file containing token sequences")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for rollout_*.pt files")
    parser.add_argument("--num-rollout", required=True, type=int, help="Number of debug rollout files to write")
    parser.add_argument("--rollout-batch-size", required=True, type=int, help="Samples in each rollout file")
    parser.add_argument(
        "--token-key",
        default="tokens",
        help="Object key containing tokens; a top-level JSON list is also accepted (default: tokens)",
    )
    parser.add_argument(
        "--expected-seq-len",
        type=int,
        default=None,
        help="Require every sequence to have this length (including the prompt-boundary token)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index assigned to the first sample (default: 0)",
    )
    return parser.parse_args()


def _read_tokens(path: Path, token_key: str, expected_seq_len: int | None) -> list[list[int]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    sequences: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    with path.open(encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            if isinstance(record, dict):
                record = record.get(token_key)
            if not isinstance(record, list) or not record:
                raise ValueError(f"{path}:{line_no}: expected a non-empty token list")
            if not all(isinstance(token, int) and not isinstance(token, bool) for token in record):
                raise ValueError(f"{path}:{line_no}: tokens must be integers")
            if any(token < 0 for token in record):
                raise ValueError(f"{path}:{line_no}: token ids must be non-negative")
            if len(record) < 2:
                raise ValueError(f"{path}:{line_no}: need at least two tokens")
            if expected_seq_len is not None and len(record) != expected_seq_len:
                raise ValueError(
                    f"{path}:{line_no}: sequence length {len(record)} != expected {expected_seq_len}"
                )

            sequence = tuple(record)
            if sequence in seen:
                raise ValueError(f"{path}:{line_no}: duplicate token sequence")
            seen.add(sequence)
            sequences.append(list(sequence))

    if not sequences:
        raise ValueError(f"{path}: no token sequences found")
    return sequences


def _make_sample(tokens: list[int], index: int) -> Sample:
    # ``get_batch`` prepends prompt_length - 1 zeros and appends one label
    # position. With response_length=len(tokens)-1 this produces exactly one
    # trailing zero at the final input position, with ones at positions that
    # predict tokens[1:].
    response_length = len(tokens) - 1
    return Sample(
        index=index,
        prompt="",
        tokens=tokens,
        response="",
        response_length=response_length,
        reward=0.0,
        loss_mask=[1] * response_length,
    )


def main() -> None:
    args = _parse_args()
    if args.num_rollout <= 0:
        raise ValueError("--num-rollout must be positive")
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout-batch-size must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")

    sequences = _read_tokens(args.input, args.token_key, args.expected_seq_len)
    needed = args.num_rollout * args.rollout_batch_size
    if len(sequences) < needed:
        raise ValueError(
            f"{args.input} contains {len(sequences)} unique sequences, but the benchmark needs {needed} "
            f"({args.num_rollout} rollouts x {args.rollout_batch_size} samples); refusing to repeat data"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rollout_id in range(args.num_rollout):
        start = rollout_id * args.rollout_batch_size
        samples = [
            _make_sample(tokens, args.start_index + start + offset)
            for offset, tokens in enumerate(sequences[start : start + args.rollout_batch_size])
        ]
        output_path = args.output_dir / f"rollout_{rollout_id}.pt"
        torch.save(
            {
                "rollout_id": rollout_id,
                "samples": [sample.to_dict() for sample in samples],
            },
            output_path,
        )

    metadata = {
        "input": str(args.input),
        "num_rollout": args.num_rollout,
        "rollout_batch_size": args.rollout_batch_size,
        "sequence_length": len(sequences[0]),
        "num_sequences_written": needed,
        "start_index": args.start_index,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
