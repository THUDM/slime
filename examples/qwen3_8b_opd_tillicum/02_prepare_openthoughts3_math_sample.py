#!/usr/bin/env python3
"""Prepare row-disjoint OpenThoughts3 math splits for slime SFT and OPD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "prompt": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "system": "system",
}


def parse_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=env.get("OT3_DATASET", "open-thoughts/OpenThoughts3-1.2M"))
    parser.add_argument("--split", default=env.get("OT3_SPLIT", "train"))
    parser.add_argument("--sft-size", type=int, default=int(env.get("SFT_SIZE", "100000")))
    parser.add_argument("--opd-size", type=int, default=int(env.get("OPD_SIZE", "50000")))
    parser.add_argument("--seed", type=int, default=int(env.get("DATA_SEED", "1234")))
    parser.add_argument("--math-field", default=env.get("DATA_MATH_FIELD", "domain"))
    parser.add_argument("--math-value", default=env.get("DATA_MATH_VALUE", "math"))
    parser.add_argument("--sft-out", default=env.get("SFT_PARQUET"))
    parser.add_argument("--opd-out", default=env.get("OPD_JSONL"))
    parser.add_argument("--metadata-out", default=env.get("SPLIT_METADATA"))
    parser.add_argument("--hf-home", default=env.get("HF_HOME"))
    parser.add_argument("--tokenizer", default=env.get("STUDENT_HF_REPO", "Qwen/Qwen3-8B-Base"))
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    missing = [name for name in ("sft_out", "opd_out", "metadata_out") if getattr(args, name) in (None, "")]
    if missing:
        parser.error(f"missing required output env/arg(s): {', '.join(missing)}")
    return args


def normalize_messages(value: Any, row: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [{"role": "user", "content": value}]

    if isinstance(value, dict) and "messages" in value:
        value = value["messages"]

    if isinstance(value, list):
        out: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            raw_role = item.get("role", item.get("from", item.get("speaker", "")))
            role = ROLE_MAP.get(str(raw_role).lower(), str(raw_role).lower())
            content = item.get("content", item.get("value", item.get("text", "")))
            if content is None:
                content = ""
            if role in {"system", "user", "assistant", "tool"}:
                out.append({"role": role, "content": str(content)})
        if out:
            return out

    prompt = first_present(row, ["prompt", "problem", "question", "instruction", "input"])
    response = first_present(row, ["response", "completion", "answer", "solution", "output"])
    if prompt is not None and response is not None:
        return [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(response)},
        ]
    if prompt is not None:
        return [{"role": "user", "content": str(prompt)}]

    raise ValueError("Could not infer OpenAI-style messages from row.")


def first_present(row: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def extract_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("messages", "conversations", "conversation"):
        if key in row and row[key] not in (None, ""):
            return normalize_messages(row[key], row)
    return normalize_messages(None, row)


def extract_prompt(row: dict[str, Any], messages: list[dict[str, str]]) -> str:
    prompt = first_present(row, ["prompt", "problem", "question", "instruction", "input"])
    if prompt is not None:
        return str(prompt)
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("Could not infer prompt text from row/messages.")


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet_rows(path: Path, rows: list[dict[str, Any]], batch_size: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            table = pa.Table.from_pylist(rows[start:end])
            if writer is None:
                writer = pq.ParquetWriter(str(path), table.schema)
            writer.write_table(table)
            if end % 5000 == 0 or end == len(rows):
                print(f"Parquet rows: {end}/{len(rows)}", flush=True)
    finally:
        if writer is not None:
            writer.close()


def write_sft_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.suffix == ".jsonl":
        write_jsonl(path, rows)
    else:
        write_parquet_rows(path, rows)


def iter_dataset_rows(ds: Dataset, indices: list[int], label: str, batch_size: int = 1000):
    selected = ds.select(indices)
    total = len(indices)
    seen = 0
    for batch in selected.iter(batch_size=batch_size):
        keys = list(batch.keys())
        if not keys:
            continue
        for row_idx in range(len(batch[keys[0]])):
            seen += 1
            if seen % 5000 == 0 or seen == total:
                print(f"{label}: {seen}/{total}", flush=True)
            yield {key: batch[key][row_idx] for key in keys}


def encode_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    encoded = tokenizer(texts, add_special_tokens=False)
    return [len(input_ids) for input_ids in encoded["input_ids"]]


def flush_token_batch(
    tokenizer: Any,
    texts: list[str],
    lengths: list[int],
    label: str,
    processed: int,
    total: int,
) -> int:
    if not texts:
        return processed
    previous = processed
    batch_size = len(texts)
    lengths.extend(encode_lengths(tokenizer, texts))
    processed += batch_size
    texts.clear()
    if processed // 5000 > previous // 5000 or processed == total:
        print(f"{label} token stats: {processed}/{total}", flush=True)
    return processed


def token_lengths(tokenizer: Any, sft_rows: list[dict[str, Any]], opd_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sft_lengths: list[int] = []
    sft_text_batch: list[str] = []
    sft_processed = 0
    for row in sft_rows:
        messages = row["messages"]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        sft_text_batch.append(text)
        if len(sft_text_batch) >= 256:
            sft_processed = flush_token_batch(
                tokenizer, sft_text_batch, sft_lengths, "SFT", sft_processed, len(sft_rows)
            )
    sft_processed = flush_token_batch(tokenizer, sft_text_batch, sft_lengths, "SFT", sft_processed, len(sft_rows))

    opd_lengths: list[int] = []
    opd_text_batch: list[str] = []
    opd_processed = 0
    for row in opd_rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = row["prompt"]
        opd_text_batch.append(text)
        if len(opd_text_batch) >= 256:
            opd_processed = flush_token_batch(
                tokenizer, opd_text_batch, opd_lengths, "OPD", opd_processed, len(opd_rows)
            )
    opd_processed = flush_token_batch(tokenizer, opd_text_batch, opd_lengths, "OPD", opd_processed, len(opd_rows))

    def stats(values: list[int]) -> dict[str, float | int]:
        return {
            "count": len(values),
            "avg": float(statistics.fmean(values)) if values else 0.0,
            "max": max(values) if values else 0,
        }

    return {
        "sft_total_token_lengths": stats(sft_lengths),
        "opd_prompt_token_lengths": stats(opd_lengths),
    }


def main() -> None:
    args = parse_args()
    sft_out = Path(args.sft_out)
    opd_out = Path(args.opd_out)
    metadata_out = Path(args.metadata_out)

    existing = [path for path in (sft_out, opd_out, metadata_out) if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "Output already exists; use --force to replace:\n" + "\n".join(f"  {path}" for path in existing)
        )

    print(f"Loading {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.hf_home)
    if args.max_source_rows is not None:
        ds = ds.select(range(min(args.max_source_rows, len(ds))))

    if args.math_field not in ds.column_names:
        raise SystemExit(
            f"Dataset does not contain math filter field {args.math_field!r}. "
            f"Columns: {', '.join(ds.column_names)}"
        )

    ds = ds.map(lambda _, idx: {"source_row_id": idx}, with_indices=True)
    math_value = str(args.math_value).lower()
    math_ds = ds.filter(lambda row: str(row.get(args.math_field, "")).lower() == math_value)

    required = args.sft_size + args.opd_size
    if len(math_ds) < required:
        raise SystemExit(f"Need {required} math rows, found {len(math_ds)} after filtering.")

    rng = random.Random(args.seed)
    shuffled = list(range(len(math_ds)))
    rng.shuffle(shuffled)
    # Split membership is seeded by the shuffled order. Materialize in dataset
    # order to avoid very slow random Arrow row reads on GPFS.
    sft_indices = sorted(shuffled[: args.sft_size])
    opd_indices = sorted(shuffled[args.sft_size : required])

    sft_rows: list[dict[str, Any]] = []
    opd_rows: list[dict[str, Any]] = []

    print(f"Building SFT rows: {len(sft_indices)}")
    for row in iter_dataset_rows(math_ds, sft_indices, "SFT rows"):
        messages = extract_messages(row)
        prompt = extract_prompt(row, messages)
        source_row_id = int(row["source_row_id"])
        sft_rows.append(
            {
                "messages": messages,
                "metadata": {
                    "source_dataset": args.dataset,
                    "source_split": args.split,
                    "source_row_id": source_row_id,
                    "prompt_sha256": prompt_hash(prompt),
                    "split": "sft",
                },
            }
        )

    print(f"Building OPD rows: {len(opd_indices)}")
    for row in iter_dataset_rows(math_ds, opd_indices, "OPD rows"):
        messages = extract_messages(row)
        prompt = extract_prompt(row, messages)
        source_row_id = int(row["source_row_id"])
        opd_rows.append(
            {
                "prompt": prompt,
                "metadata": {
                    "source_dataset": args.dataset,
                    "source_split": args.split,
                    "source_row_id": source_row_id,
                    "prompt_sha256": prompt_hash(prompt),
                    "split": "opd",
                },
            }
        )

    sft_source_ids = [row["metadata"]["source_row_id"] for row in sft_rows]
    opd_source_ids = [row["metadata"]["source_row_id"] for row in opd_rows]
    overlap = sorted(set(sft_source_ids).intersection(opd_source_ids))
    if overlap:
        raise RuntimeError(f"SFT/OPD row split overlap detected: first overlaps {overlap[:10]}")

    write_sft_rows(sft_out, sft_rows)
    write_jsonl(opd_out, opd_rows)
    print(f"Wrote {sft_out}")
    print(f"Wrote {opd_out}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.hf_home, trust_remote_code=True)
    length_stats = token_lengths(tokenizer, sft_rows, opd_rows)

    metadata = {
        "seed": args.seed,
        "dataset": args.dataset,
        "source_split": args.split,
        "filtering_criteria": {
            "field": args.math_field,
            "value": args.math_value,
            "operation": "case-insensitive equality",
        },
        "sample_materialization": "Seeded split membership, indices sorted within each split before row materialization.",
        "counts": {
            "source_rows_seen": len(ds),
            "math_rows_after_filter": len(math_ds),
            "sft": len(sft_rows),
            "opd": len(opd_rows),
        },
        "outputs": {
            "sft_data": str(sft_out),
            "opd_jsonl": str(opd_out),
        },
        "source_row_ids": {
            "sft": sft_source_ids,
            "opd": opd_source_ids,
        },
        "prompt_hashes": {
            "sft": [row["metadata"]["prompt_sha256"] for row in sft_rows],
            "opd": [row["metadata"]["prompt_sha256"] for row in opd_rows],
        },
        "tokenizer": args.tokenizer,
        "token_lengths": length_stats,
    }

    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {metadata_out}")


if __name__ == "__main__":
    main()
