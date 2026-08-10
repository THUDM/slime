#!/usr/bin/env python3
"""Compare matching Megatron and SGLang decoder-layer outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_LAYER_KEY_RE = re.compile(r"(?:^|\.)layers\.(\d+)$")
_NON_ROLLOUT_REQUEST_PREFIXES = ("HEALTH_CHECK_",)


@dataclass
class TrainSequence:
    tokens: torch.Tensor
    layers: dict[int, torch.Tensor]
    source: str


def _load_records(path: Path) -> Iterator[dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        yield from value
        return
    raise TypeError(f"Unsupported layer dump payload in {path}: {type(value)}")


def _find_suffix(record: dict[str, Any], suffix: str, *, required: bool = True):
    matches = [value for key, value in record.items() if key.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    raise KeyError(f"Expected one key ending in {suffix!r}, found {len(matches)}")


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if tensors:
            return tensors[0]
    raise TypeError(f"Expected a tensor layer output, got {type(value)}")


def _token_rows(value: Any, num_tokens: int, context: str) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor.ndim < 2:
        raise ValueError(f"{context} must have a hidden dimension, got {tensor.shape}")
    rows = tensor.reshape(-1, tensor.shape[-1])
    if rows.shape[0] != num_tokens:
        raise ValueError(f"{context} has {rows.shape[0]} token rows, expected {num_tokens}")
    return rows


def _sglang_layer_token_rows(value: Any, num_tokens: int, context: str) -> torch.Tensor:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        raise TypeError(f"{context} must contain the SGLang layer delta and residual tensors")
    delta, residual = value[:2]
    if not isinstance(delta, torch.Tensor) or not isinstance(residual, torch.Tensor):
        raise TypeError(f"{context} contains non-tensor layer outputs")
    if delta.shape != residual.shape or delta.dtype != residual.dtype:
        raise ValueError(
            f"{context} delta/residual mismatch: " f"{delta.shape}/{delta.dtype} != {residual.shape}/{residual.dtype}"
        )
    layer_output = (delta.float() + residual.float()).to(delta.dtype)
    return _token_rows(layer_output, num_tokens, context)


def _layer_outputs(record: dict[str, Any], selected_layers: set[int]):
    outputs = {}
    for key, value in record.items():
        match = _LAYER_KEY_RE.search(key)
        if match is None:
            continue
        layer_id = int(match.group(1))
        if layer_id in selected_layers:
            outputs[layer_id] = value
    return outputs


def load_train_sequences(dump_dir: Path, selected_layers: set[int]) -> list[TrainSequence]:
    dump_files = sorted(dump_dir.glob("rank*/actor_Pass*.pt"))
    if not dump_files:
        raise FileNotFoundError(f"No Megatron layer dumps found under {dump_dir}")

    sequences = []
    for dump_file in dump_files:
        for record in _load_records(dump_file):
            tokens = record["input_ids"].reshape(-1).to(torch.int64)
            cu_seqlens = record["cu_seqlens"].reshape(-1).tolist()
            layer_values = record.get("layers", {})
            missing = selected_layers - set(layer_values)
            if missing:
                raise KeyError(f"{dump_file} is missing Megatron layers {sorted(missing)}")
            layer_rows = {
                layer_id: _token_rows(layer_values[layer_id], len(tokens), f"{dump_file}:layer{layer_id}")
                for layer_id in selected_layers
            }
            for sequence_index, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:], strict=False)):
                sequence_tokens = tokens[start:end]
                if sequence_tokens.numel() == 0 or torch.count_nonzero(sequence_tokens) == 0:
                    continue
                sequences.append(
                    TrainSequence(
                        tokens=sequence_tokens,
                        layers={layer_id: rows[start:end] for layer_id, rows in layer_rows.items()},
                        source=f"{dump_file}:sequence{sequence_index}",
                    )
                )
    if not sequences:
        raise RuntimeError("Megatron dumps contained no non-padding sequences")
    return sequences


def _sglang_segments(record: dict[str, Any]):
    input_ids = _find_suffix(record, ".forward_batch_info.input_ids").reshape(-1)
    positions = _find_suffix(record, ".forward_batch_info.positions").reshape(-1)
    rids = _find_suffix(record, ".forward_batch_info.rids")
    if not rids:
        return input_ids, positions, []

    extend_seq_lens = _find_suffix(record, ".forward_batch_info.extend_seq_lens", required=False)
    if extend_seq_lens is None:
        counts = [1] * len(rids)
    else:
        counts = [int(value) for value in extend_seq_lens.reshape(-1).tolist()]
    if len(counts) != len(rids) or sum(counts) != input_ids.numel():
        raise ValueError(
            "SGLang request segmentation mismatch: " f"rids={len(rids)}, counts={counts}, tokens={input_ids.numel()}"
        )

    segments = []
    start = 0
    for rid, count in zip(rids, counts, strict=True):
        end = start + count
        rid = str(rid)
        if count and not rid.startswith(_NON_ROLLOUT_REQUEST_PREFIXES):
            segments.append((rid, slice(start, end)))
        start = end
    return input_ids.to(torch.int64), positions.to(torch.int64), segments


def _sglang_dump_files(dump_dir: Path) -> list[Path]:
    process_dirs = sorted(path for path in dump_dir.iterdir() if path.is_dir())
    files = []
    for process_dir in process_dirs:
        files.extend(sorted(process_dir.glob("Chunk*.pt")))
        files.extend(sorted(process_dir.glob("Pass*.pt")))
    if not files:
        raise FileNotFoundError(f"No SGLang layer dumps found under {dump_dir}")
    return files


def map_requests_to_train_sequences(dump_files: list[Path], train_sequences: list[TrainSequence]) -> dict[str, int]:
    observations: dict[str, dict[int, int]] = {}
    for dump_file in dump_files:
        for record in _load_records(dump_file):
            input_ids, positions, segments = _sglang_segments(record)
            for rid, token_slice in segments:
                request_observations = observations.setdefault(rid, {})
                for position, token_id in zip(
                    positions[token_slice].tolist(),
                    input_ids[token_slice].tolist(),
                    strict=True,
                ):
                    previous = request_observations.setdefault(position, token_id)
                    if previous != token_id:
                        raise ValueError(
                            f"SGLang request {rid} changed token at position {position}: " f"{previous} != {token_id}"
                        )

    mapping = {}
    for rid, request_observations in observations.items():
        candidates = []
        for sequence_id, sequence in enumerate(train_sequences):
            if all(
                0 <= position < sequence.tokens.numel() and int(sequence.tokens[position]) == token_id
                for position, token_id in request_observations.items()
            ):
                candidates.append(sequence_id)
        if not candidates:
            raise RuntimeError(f"Could not map SGLang request {rid} to any Megatron token sequence")
        mapping[rid] = candidates[0]
    if not mapping:
        raise RuntimeError("SGLang dumps contained no request observations")
    return mapping


def compare_layer_outputs(
    dump_files: list[Path],
    train_sequences: list[TrainSequence],
    request_mapping: dict[str, int],
    selected_layers: set[int],
):
    stats = {layer_id: {"max_abs": 0.0, "sum_abs": 0.0, "numel": 0, "tokens": 0} for layer_id in selected_layers}
    compared: set[tuple[str, int, int]] = set()

    for dump_file in dump_files:
        for record in _load_records(dump_file):
            input_ids, positions, segments = _sglang_segments(record)
            if not segments:
                continue
            rollout_layers = _layer_outputs(record, selected_layers)
            missing = selected_layers - set(rollout_layers)
            if missing:
                raise KeyError(f"{dump_file} is missing SGLang layers {sorted(missing)}")
            rollout_rows = {
                layer_id: _sglang_layer_token_rows(
                    value,
                    input_ids.numel(),
                    f"{dump_file}:layer{layer_id}",
                )
                for layer_id, value in rollout_layers.items()
            }

            for rid, token_slice in segments:
                train_sequence = train_sequences[request_mapping[rid]]
                segment_positions = positions[token_slice]
                for layer_id in selected_layers:
                    keep = torch.tensor(
                        [
                            # A causal LM never consumes the hidden state at the
                            # final input position to score a token in this
                            # sequence. SGLang may still execute that terminal
                            # token after sampling it, whereas Megatron's
                            # log-prob forward stops at score-producing
                            # positions. The terminal state therefore has no
                            # corresponding training row to compare.
                            int(position) < train_sequence.tokens.numel() - 1
                            and (rid, int(position), layer_id) not in compared
                            for position in segment_positions.tolist()
                        ],
                        dtype=torch.bool,
                    )
                    if not torch.any(keep):
                        continue
                    kept_positions = segment_positions[keep]
                    if torch.any(kept_positions < 0) or torch.any(kept_positions >= train_sequence.tokens.numel()):
                        raise IndexError(f"SGLang request {rid} contains positions outside its " "Megatron sequence")
                    rollout_value = rollout_rows[layer_id][token_slice][keep]
                    train_value = train_sequence.layers[layer_id][kept_positions]
                    difference = (rollout_value.float() - train_value.float()).abs()
                    layer_stats = stats[layer_id]
                    layer_stats["max_abs"] = max(layer_stats["max_abs"], float(difference.max().item()))
                    layer_stats["sum_abs"] += float(difference.sum().item())
                    layer_stats["numel"] += difference.numel()
                    layer_stats["tokens"] += kept_positions.numel()
                    for position in kept_positions.tolist():
                        compared.add((rid, int(position), layer_id))

    for layer_stats in stats.values():
        layer_stats["mean_abs"] = (
            layer_stats.pop("sum_abs") / layer_stats["numel"] if layer_stats["numel"] else float("nan")
        )
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--sglang-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--max-hidden-diff", type=float, default=1e-7)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads <= 0:
        raise ValueError(f"--num-threads must be positive, got {args.num_threads}")
    # Records contain only a few token rows. Launching the host-wide PyTorch
    # thread pool for every small reduction costs far more than the arithmetic.
    torch.set_num_threads(args.num_threads)
    selected_layers = set(args.layers)
    train_sequences = load_train_sequences(args.megatron_dir, selected_layers)
    dump_files = _sglang_dump_files(args.sglang_dir)
    request_mapping = map_requests_to_train_sequences(dump_files, train_sequences)
    stats = compare_layer_outputs(
        dump_files,
        train_sequences,
        request_mapping,
        selected_layers,
    )

    failed = False
    for layer_id in sorted(stats):
        layer_stats = stats[layer_id]
        print(
            f"layer={layer_id} tokens={layer_stats['tokens']} "
            f"max_abs={layer_stats['max_abs']:.12g} "
            f"mean_abs={layer_stats['mean_abs']:.12g}"
        )
        if not layer_stats["tokens"] or layer_stats["max_abs"] > args.max_hidden_diff:
            failed = True

    result = {
        "max_hidden_diff": args.max_hidden_diff,
        "request_mapping": request_mapping,
        "layers": stats,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
