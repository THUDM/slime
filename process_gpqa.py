#!/usr/bin/env python3
"""Convert GPQA CSV exports into Slime eval JSONL format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GPQA CSV into eval-ready JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Path to GPQA CSV file.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed that is combined with each record id to shuffle choices.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable answer choice shuffling (keeps correct option at letter A).",
    )
    return parser.parse_args()


def _clean_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _format_prompt(question: str, choices: Dict[str, str]) -> List[Dict[str, str]]:
    lines: List[str] = [_clean_text(question), "", "Options:"]
    for letter, choice in choices.items():
        lines.append(f"{letter}. {choice}")
    lines.extend(["", "Answer with the single letter of the correct choice."])
    return [{"role": "user", "content": "\n".join(lines)}]


def _per_record_seed(base_seed: int, record_id: str) -> int:
    digest = hashlib.md5(record_id.encode("utf-8")).hexdigest()
    return base_seed + int(digest, 16)


def convert(input_path: Path, output_path: Path, *, seed: int, shuffle: bool) -> None:
    required_columns = [
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    ]
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as jsonl_file:
            for index, row in enumerate(reader):
                question = row["Question"]
                correct = row["Correct Answer"]
                incorrects = [
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ]

                if not question or not correct or any(answer is None for answer in incorrects):
                    continue

                option_pool: List[str] = [_clean_text(correct)] + [_clean_text(opt) for opt in incorrects]
                record_id = row.get("Record ID") or f"{input_path.stem}-{index}"

                indices: List[int] = list(range(len(option_pool)))
                if shuffle:
                    per_record = _per_record_seed(seed, record_id)
                    # Deterministically shuffle without mutating the shared generator state.
                    for i in range(len(indices) - 1, 0, -1):
                        per_record = (per_record * 6364136223846793005 + 1) & ((1 << 64) - 1)
                        j = per_record % (i + 1)
                        indices[i], indices[j] = indices[j], indices[i]

                letters = [chr(ord("A") + i) for i in range(len(indices))]
                choices: Dict[str, str] = {}
                correct_letter = None
                for letter, option_index in zip(letters, indices):
                    text = option_pool[option_index]
                    choices[letter] = text
                    if option_index == 0:
                        correct_letter = letter

                if correct_letter is None:
                    # Skip malformed rows where the correct answer vanished.
                    continue

                prompt = _format_prompt(question, choices)
                metadata = {
                    "choices": choices,
                    "correct_letter": correct_letter,
                    "correct_answer": option_pool[0],
                    "valid_letters": letters,
                    "record_id": record_id,
                    "source": "gpqa_diamond",
                    "rm_type": "gpqa",
                }

                json_record = {
                    "prompt": prompt,
                    "label": correct_letter,
                    "metadata": metadata,
                }
                jsonl_file.write(json.dumps(json_record, ensure_ascii=False) + "\n")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args()
    convert(args.input, args.output, seed=args.seed, shuffle=not args.no_shuffle)


if __name__ == "__main__":
    main()
