"""Benchmark trainer-to-engine weight synchronization policies.

The benchmark is intentionally independent of a model and Ray/SGLang lifecycle.
Rank 0 acts as the trainer; the other ranks are partitioned into one or more
rollout-engine groups.  It measures the data-plane behavior of bucket credits,
engine waves, and phase strides without changing slime's production updater.

Examples:

    torchrun --standalone --nproc-per-node=4 tools/benchmark_weight_sync.py \
        --transports nccl_broadcast p2p \
        --buffer-bytes 16MiB 64MiB \
        --max-inflight-buckets 1 2 \
        --engine-wave-policies all_at_once serialized windowed \
        --output-json /tmp/weight-sync.json

The layout is derived from the launched world size.  For example, a four-rank
run with ``--engine-group-sizes 1 2`` assigns rank 1 and ranks 2-3 to two
heterogeneous rollout engines.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import os
import random
import socket
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class EngineGroup:
    engine_id: int
    ranks: tuple[int, ...]


@dataclass(frozen=True)
class TransferTask:
    bucket_id: int
    engine_id: int
    message_bytes: int
    transport: str = "nccl_broadcast"


@dataclass(frozen=True)
class ExperimentConfig:
    transport: str
    message_bytes: int
    max_inflight_buckets: int
    max_inflight_bytes: int
    max_inflight_engine_groups: int
    engine_wave_policy: str
    phase_stride_us: int


@dataclass
class PendingTransfer:
    task: TransferTask
    tensor: torch.Tensor
    load_tensor: torch.Tensor | None
    transfer_start: torch.cuda.Event | None
    transfer_end: torch.cuda.Event | None
    load_start: torch.cuda.Event | None
    load_end: torch.cuda.Event | None
    api_launch_timestamp_ns: int
    control_wait_ms: float
    host_transfer_ms: float | None = None
    host_load_ms: float | None = None


SIZE_SUFFIXES = {
    "b": 1,
    "kib": 1 << 10,
    "mib": 1 << 20,
    "gib": 1 << 30,
    "kb": 1_000,
    "mb": 1_000_000,
    "gb": 1_000_000_000,
}


def parse_size(value: str) -> int:
    """Parse an integer byte count with an optional binary or decimal suffix."""
    normalized = value.strip().lower().replace("_", "")
    for suffix in sorted(SIZE_SUFFIXES, key=len, reverse=True):
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)]
            result = float(number) * SIZE_SUFFIXES[suffix]
            if not result.is_integer() or result <= 0:
                raise argparse.ArgumentTypeError(f"invalid positive byte size: {value}")
            return int(result)
    try:
        result = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError(f"invalid positive byte size: {value}")
    return result


def parse_byte_credit(value: str) -> int:
    """Parse an optional byte limit, where zero means derive the limit."""
    if value.strip().replace("_", "") == "0":
        return 0
    return parse_size(value)


def build_engine_groups(
    world_size: int,
    *,
    engine_group_size: int = 1,
    engine_group_sizes: Sequence[int] | None = None,
) -> list[EngineGroup]:
    """Partition all non-trainer ranks into rollout-engine groups."""
    if world_size < 2:
        raise ValueError(f"world_size must be at least 2, got {world_size}")
    available_ranks = world_size - 1
    if engine_group_sizes is None:
        if engine_group_size < 1:
            raise ValueError("engine_group_size must be positive")
        if available_ranks % engine_group_size != 0:
            raise ValueError(
                f"{available_ranks} engine ranks are not divisible by engine_group_size "
                f"{engine_group_size}; pass --engine-group-sizes for a heterogeneous layout"
            )
        sizes = [engine_group_size] * (available_ranks // engine_group_size)
    else:
        sizes = list(engine_group_sizes)
        if not sizes or any(size < 1 for size in sizes):
            raise ValueError("engine_group_sizes must contain positive values")
        if sum(sizes) != available_ranks:
            raise ValueError(f"engine_group_sizes sum to {sum(sizes)}, expected {available_ranks}")

    groups = []
    first_rank = 1
    for engine_id, size in enumerate(sizes):
        groups.append(EngineGroup(engine_id=engine_id, ranks=tuple(range(first_rank, first_rank + size))))
        first_rank += size
    return groups


def build_transfer_tasks(
    bucket_count: int,
    engine_groups: Sequence[EngineGroup],
    message_bytes: int,
    transport: str = "nccl_broadcast",
) -> list[TransferTask]:
    """Build bucket-major trainer-to-engine transfers."""
    if bucket_count < 1:
        raise ValueError("bucket_count must be positive")
    return [
        TransferTask(
            bucket_id=bucket_id,
            engine_id=engine_group.engine_id,
            message_bytes=message_bytes,
            transport=transport,
        )
        for bucket_id in range(bucket_count)
        for engine_group in engine_groups
    ]


def _wave_fits(
    wave: Sequence[TransferTask],
    task: TransferTask,
    *,
    max_buckets: int,
    max_bytes: int,
    max_engine_groups: int,
) -> bool:
    candidate = [*wave, task]
    bucket_ids = {item.bucket_id for item in candidate}
    engine_ids = {item.engine_id for item in candidate}
    inflight_bytes = sum(
        next(item.message_bytes for item in candidate if item.bucket_id == bucket_id) for bucket_id in bucket_ids
    )
    return len(bucket_ids) <= max_buckets and inflight_bytes <= max_bytes and len(engine_ids) <= max_engine_groups


def plan_transfer_waves(
    tasks: Sequence[TransferTask],
    *,
    policy: str,
    max_inflight_buckets: int,
    max_inflight_bytes: int,
    max_inflight_engine_groups: int,
) -> list[list[TransferTask]]:
    """Pack transfers into deterministic waves under bucket/byte/group credits."""
    if not tasks:
        return []
    if policy not in {"all_at_once", "serialized", "windowed"}:
        raise ValueError(f"unsupported engine wave policy: {policy}")
    if max_inflight_buckets < 1 or max_inflight_engine_groups < 1:
        raise ValueError("inflight bucket and engine-group limits must be positive")
    largest_bucket = max(task.message_bytes for task in tasks)
    effective_max_bytes = max_inflight_bytes or largest_bucket * max_inflight_buckets
    if effective_max_bytes < largest_bucket:
        raise ValueError(
            f"max_inflight_bytes {effective_max_bytes} is smaller than one bucket " f"({largest_bucket} bytes)"
        )

    engine_count = len({task.engine_id for task in tasks})
    if policy == "serialized":
        effective_max_buckets = 1
        effective_max_engine_groups = 1
    elif policy == "all_at_once":
        effective_max_buckets = max_inflight_buckets
        effective_max_engine_groups = engine_count
    else:
        effective_max_buckets = max_inflight_buckets
        effective_max_engine_groups = max_inflight_engine_groups

    waves: list[list[TransferTask]] = []
    current: list[TransferTask] = []
    for task in tasks:
        if current and not _wave_fits(
            current,
            task,
            max_buckets=effective_max_buckets,
            max_bytes=effective_max_bytes,
            max_engine_groups=effective_max_engine_groups,
        ):
            waves.append(current)
            current = []
        current.append(task)
    if current:
        waves.append(current)
    return waves


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile, or ``None`` for no samples."""
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _summarize(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def _resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    return "nccl" if torch.cuda.is_available() else "gloo"


def _group_for_rank(engine_groups: Sequence[EngineGroup], rank: int) -> EngineGroup | None:
    matches = [group for group in engine_groups if rank in group.ranks]
    if len(matches) > 1:
        raise ValueError(f"rank {rank} belongs to multiple engine groups")
    return matches[0] if matches else None


def _create_engine_process_groups(
    engine_groups: Sequence[EngineGroup], backend: str
) -> dict[int, dist.ProcessGroup | None]:
    process_groups = {}
    for engine_group in engine_groups:
        process_group = dist.new_group(ranks=[0, *engine_group.ranks], backend=backend)
        process_groups[engine_group.engine_id] = (
            None if process_group == dist.GroupMember.NON_GROUP_MEMBER else process_group
        )
    return process_groups


def _wait_until_ns(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 100_000:
            time.sleep(0)


def _launch_transfer(
    task: TransferTask,
    *,
    rank: int,
    engine_groups_by_id: dict[int, EngineGroup],
    process_groups: dict[int, dist.ProcessGroup | None],
    control_process_groups: dict[int, dist.ProcessGroup | None],
    streams: dict[int, torch.cuda.Stream | None],
    trainer_tensors: dict[int, torch.Tensor],
    receiver_tensors: dict[int, torch.Tensor],
    load_tensors: dict[int, torch.Tensor],
    simulate_load: bool,
) -> PendingTransfer | None:
    engine_group = engine_groups_by_id[task.engine_id]
    is_trainer = rank == 0
    is_receiver = rank in engine_group.ranks
    if not is_trainer and not is_receiver:
        return None

    control_process_group = control_process_groups[task.engine_id]
    if control_process_group is None:
        raise RuntimeError(f"rank {rank} is not a control-group member for engine {task.engine_id}")
    control_start_ns = time.perf_counter_ns()
    dist.barrier(group=control_process_group)
    control_wait_ms = (time.perf_counter_ns() - control_start_ns) / 1_000_000

    if is_trainer:
        tensor = trainer_tensors[task.bucket_id]
    else:
        tensor = receiver_tensors[task.bucket_id]
    stream = streams[task.engine_id]
    transfer_start = torch.cuda.Event(enable_timing=True) if stream is not None else None
    transfer_end = torch.cuda.Event(enable_timing=True) if stream is not None else None
    load_start = torch.cuda.Event(enable_timing=True) if stream is not None and is_receiver and simulate_load else None
    load_end = torch.cuda.Event(enable_timing=True) if stream is not None and is_receiver and simulate_load else None
    load_tensor = load_tensors[task.bucket_id] if is_receiver and simulate_load else None
    launch_ns = time.perf_counter_ns()
    host_start_ns = launch_ns
    host_load_ms = None

    def issue() -> None:
        nonlocal host_load_ms
        if transfer_start is not None:
            transfer_start.record(stream)
        if task.transport == "nccl_broadcast":
            process_group = process_groups[task.engine_id]
            if process_group is None:
                raise RuntimeError(f"rank {rank} is not a member of engine group {task.engine_id}")
            work = dist.broadcast(tensor, src=0, group=process_group, async_op=True)
            work.wait()
        elif is_trainer:
            works = [dist.isend(tensor, dst=target_rank) for target_rank in engine_group.ranks]
            for work in works:
                work.wait()
        else:
            dist.irecv(tensor, src=0).wait()
        if transfer_end is not None:
            transfer_end.record(stream)
        if load_tensor is not None and load_start is not None:
            assert load_end is not None
            load_start.record(stream)
            load_tensor.copy_(tensor, non_blocking=True)
            load_end.record(stream)
        elif load_tensor is not None:
            load_start_ns = time.perf_counter_ns()
            load_tensor.copy_(tensor)
            host_load_ms = (time.perf_counter_ns() - load_start_ns) / 1_000_000

    if stream is not None:
        with torch.cuda.stream(stream):
            issue()
        host_transfer_ms = None
    else:
        issue()
        host_transfer_ms = (time.perf_counter_ns() - host_start_ns) / 1_000_000

    return PendingTransfer(
        task=task,
        tensor=tensor,
        load_tensor=load_tensor,
        transfer_start=transfer_start,
        transfer_end=transfer_end,
        load_start=load_start,
        load_end=load_end,
        api_launch_timestamp_ns=launch_ns,
        control_wait_ms=control_wait_ms,
        host_transfer_ms=host_transfer_ms,
        host_load_ms=host_load_ms,
    )


def _prepare_tensors(
    *,
    bucket_count: int,
    message_bytes: int,
    rank: int,
    local_engine_group: EngineGroup | None,
    device: torch.device,
    simulate_load: bool,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Allocate reusable buffers outside the measured iteration interval."""
    trainer_tensors = (
        {bucket_id: torch.empty(message_bytes, dtype=torch.uint8, device=device) for bucket_id in range(bucket_count)}
        if rank == 0
        else {}
    )
    receiver_tensors = (
        {bucket_id: torch.empty(message_bytes, dtype=torch.uint8, device=device) for bucket_id in range(bucket_count)}
        if local_engine_group is not None
        else {}
    )
    load_tensors = (
        {bucket_id: torch.empty_like(tensor) for bucket_id, tensor in receiver_tensors.items()}
        if simulate_load
        else {}
    )
    return trainer_tensors, receiver_tensors, load_tensors


def _complete_pending(
    pending: Sequence[PendingTransfer],
    *,
    rank: int,
    iteration: int,
    wave_index: int,
    validate: bool,
    value_seed: int,
) -> list[dict[str, Any]]:
    records = []
    for item in pending:
        final_event = item.load_end if item.load_end is not None else item.transfer_end
        if final_event is not None:
            final_event.synchronize()
        completion_observed_ns = time.perf_counter_ns()
        if item.transfer_start is not None:
            assert item.transfer_end is not None
            transfer_ms = item.transfer_start.elapsed_time(item.transfer_end)
        else:
            assert item.host_transfer_ms is not None
            transfer_ms = item.host_transfer_ms
        if item.load_start is not None:
            assert item.load_end is not None
            load_ms = item.load_start.elapsed_time(item.load_end)
        else:
            load_ms = item.host_load_ms

        if validate and rank != 0:
            expected = (value_seed + item.task.bucket_id) % 251 + 1
            tensor = item.load_tensor if item.load_tensor is not None else item.tensor
            if int(tensor[0].item()) != expected or int(tensor[-1].item()) != expected:
                raise RuntimeError(
                    f"weight payload mismatch for bucket {item.task.bucket_id}, "
                    f"engine {item.task.engine_id}, rank {rank}"
                )

        role = "trainer" if rank == 0 else "engine"
        records.append(
            {
                "iteration": iteration,
                "weight_version": iteration,
                "wave_index": wave_index,
                "bucket_id": item.task.bucket_id,
                "engine_id": item.task.engine_id,
                "rank": rank,
                "role": role,
                "operation": "weight_bucket_send" if rank == 0 else "engine_bucket_receive",
                "transport": item.task.transport,
                "message_bytes": item.task.message_bytes,
                "api_launch_timestamp_ns": item.api_launch_timestamp_ns,
                "control_wait_ms": item.control_wait_ms,
                "completion_observed_timestamp_ns": completion_observed_ns,
                "transfer_ms": transfer_ms,
                "synthetic_load_ms": load_ms,
            }
        )
    return records


def run_iteration(
    config: ExperimentConfig,
    waves: Sequence[Sequence[TransferTask]],
    *,
    rank: int,
    local_engine_group: EngineGroup | None,
    engine_groups_by_id: dict[int, EngineGroup],
    process_groups: dict[int, dist.ProcessGroup | None],
    control_process_groups: dict[int, dist.ProcessGroup | None],
    streams: dict[int, torch.cuda.Stream | None],
    control_group: dist.ProcessGroup,
    device: torch.device,
    trainer_tensors: dict[int, torch.Tensor],
    receiver_tensors: dict[int, torch.Tensor],
    load_tensors: dict[int, torch.Tensor],
    iteration: int,
    simulate_load: bool,
    validate: bool,
) -> tuple[float, list[dict[str, Any]]]:
    """Execute one complete weight version and return local trace records."""
    value_seed = iteration * 17
    for bucket_id, tensor in trainer_tensors.items():
        tensor.fill_((value_seed + bucket_id) % 251 + 1)
    if device.type == "cuda" and trainer_tensors:
        torch.cuda.synchronize(device)
    dist.barrier(group=control_group)
    iteration_start_ns = time.perf_counter_ns()
    records = []
    for wave_index, wave in enumerate(waves):
        wave_anchor_ns = time.perf_counter_ns()
        engine_slots = {engine_id: slot for slot, engine_id in enumerate(sorted({task.engine_id for task in wave}))}
        pending = []
        for task in sorted(wave, key=lambda item: (engine_slots[item.engine_id], item.bucket_id)):
            _wait_until_ns(wave_anchor_ns + engine_slots[task.engine_id] * config.phase_stride_us * 1_000)
            item = _launch_transfer(
                task,
                rank=rank,
                engine_groups_by_id=engine_groups_by_id,
                process_groups=process_groups,
                control_process_groups=control_process_groups,
                streams=streams,
                trainer_tensors=trainer_tensors,
                receiver_tensors=receiver_tensors,
                load_tensors=load_tensors,
                simulate_load=simulate_load,
            )
            if item is not None:
                pending.append(item)
        records.extend(
            _complete_pending(
                pending,
                rank=rank,
                iteration=iteration,
                wave_index=wave_index,
                validate=validate,
                value_seed=value_seed,
            )
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - iteration_start_ns) / 1_000_000
    dist.barrier(group=control_group)
    return elapsed_ms, records


def _aggregate_experiment(
    config: ExperimentConfig,
    waves: Sequence[Sequence[TransferTask]],
    gathered: Sequence[dict[str, Any]],
    *,
    engine_groups: Sequence[EngineGroup],
) -> dict[str, Any]:
    payloads_by_rank = {item["rank"]: item for item in gathered}
    trainer_payload = payloads_by_rank[0]
    trainer_total_ms = trainer_payload["iteration_ms"]
    total_ms = [
        max(item["iteration_ms"][iteration] for item in gathered) for iteration in range(len(trainer_total_ms))
    ]
    records = [record for item in gathered for record in item["records"]]
    trainer_records = [record for record in records if record["role"] == "trainer"]
    engine_records = [record for record in records if record["role"] == "engine"]

    trainer_busy_by_iteration: dict[int, float] = {}
    engine_busy_by_rank_iteration: dict[tuple[int, int], float] = {}
    for record in trainer_records:
        trainer_busy_by_iteration.setdefault(record["iteration"], 0.0)
        trainer_busy_by_iteration[record["iteration"]] += record["transfer_ms"]
    for record in engine_records:
        key = (record["rank"], record["iteration"])
        engine_busy_by_rank_iteration.setdefault(key, 0.0)
        engine_busy_by_rank_iteration[key] += record["transfer_ms"] + (record["synthetic_load_ms"] or 0.0)

    trainer_idle = [
        max(0.0, duration - trainer_busy_by_iteration.get(iteration, 0.0))
        for iteration, duration in enumerate(trainer_total_ms)
    ]
    engine_idle = [
        max(0.0, payloads_by_rank[rank]["iteration_ms"][iteration] - busy)
        for (rank, iteration), busy in engine_busy_by_rank_iteration.items()
    ]

    by_transfer: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["iteration"], record["bucket_id"], record["engine_id"])
        by_transfer.setdefault(key, []).append(record)
    hostnames = {item.get("hostname") for item in gathered}
    cross_rank_clock_comparable = None not in hostnames and len(hostnames) == 1
    rank_start_skew_us = []
    engine_finish_skew_us = []
    if cross_rank_clock_comparable:
        for transfer_records in by_transfer.values():
            launches = [record["api_launch_timestamp_ns"] for record in transfer_records]
            rank_start_skew_us.append((max(launches) - min(launches)) / 1_000)
            receiver_finishes = [
                record["api_launch_timestamp_ns"] + record["transfer_ms"] * 1_000_000
                for record in transfer_records
                if record["role"] == "engine"
            ]
            if receiver_finishes:
                engine_finish_skew_us.append((max(receiver_finishes) - min(receiver_finishes)) / 1_000)

    max_inflight_bytes = max(
        sum(
            next(task.message_bytes for task in wave if task.bucket_id == bucket_id)
            for bucket_id in {task.bucket_id for task in wave}
        )
        for wave in waves
    )
    max_inflight_wire_bytes = max(
        sum(task.message_bytes * len(engine_groups[task.engine_id].ranks) for task in wave) for wave in waves
    )
    max_inflight_engine_groups = max(len({task.engine_id for task in wave}) for wave in waves)
    result = {
        **asdict(config),
        "wave_count": len(waves),
        "max_inflight_bytes_observed": max_inflight_bytes,
        "max_inflight_wire_bytes_observed": max_inflight_wire_bytes,
        "max_inflight_engine_groups_observed": max_inflight_engine_groups,
        "cross_rank_clock_comparable": cross_rank_clock_comparable,
        "weight_sync_total_ms": _summarize(total_ms),
        "trainer_total_ms": _summarize(trainer_total_ms),
        "bucket_send_ms": _summarize([record["transfer_ms"] for record in trainer_records]),
        "engine_receive_ms": _summarize([record["transfer_ms"] for record in engine_records]),
        "engine_load_ms": _summarize(
            [record["synthetic_load_ms"] for record in engine_records if record["synthetic_load_ms"] is not None]
        ),
        "trainer_control_wait_ms": _summarize([record["control_wait_ms"] for record in trainer_records]),
        "engine_control_wait_ms": _summarize([record["control_wait_ms"] for record in engine_records]),
        "trainer_idle_ms": _summarize(trainer_idle),
        "engine_idle_ms": _summarize(engine_idle),
        "rank_start_skew_us": _summarize(rank_start_skew_us),
        "engine_finish_skew_us": _summarize(engine_finish_skew_us),
        "samples": len(total_ms),
    }
    return result


def _build_experiment_configs(args: argparse.Namespace, engine_count: int) -> list[ExperimentConfig]:
    configs = []
    for values in itertools.product(
        args.transports,
        args.buffer_bytes,
        args.max_inflight_buckets,
        args.max_inflight_bytes,
        args.engine_wave_policies,
        args.phase_stride_us,
    ):
        transport, message_bytes, max_buckets, max_bytes, policy, stride_us = values
        max_groups = min(args.max_inflight_engine_groups, engine_count)
        configs.append(
            ExperimentConfig(
                transport=transport,
                message_bytes=message_bytes,
                max_inflight_buckets=max_buckets,
                max_inflight_bytes=max_bytes,
                max_inflight_engine_groups=max_groups,
                engine_wave_policy=policy,
                phase_stride_us=stride_us,
            )
        )
    random.Random(args.seed).shuffle(configs)
    return configs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument(
        "--transports",
        nargs="+",
        choices=("nccl_broadcast", "p2p"),
        default=["nccl_broadcast"],
    )
    parser.add_argument("--buffer-bytes", nargs="+", type=parse_size, default=[parse_size("64MiB")])
    parser.add_argument("--max-inflight-buckets", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--max-inflight-bytes",
        nargs="+",
        type=parse_byte_credit,
        default=[0],
        help="0 derives the byte credit from buffer size and max-inflight-buckets",
    )
    parser.add_argument(
        "--engine-wave-policies",
        nargs="+",
        choices=("all_at_once", "serialized", "windowed"),
        default=["all_at_once", "serialized", "windowed"],
    )
    parser.add_argument("--max-inflight-engine-groups", type=int, default=2)
    parser.add_argument("--phase-stride-us", nargs="+", type=int, default=[0])
    parser.add_argument("--engine-group-size", type=int, default=1)
    parser.add_argument("--engine-group-sizes", nargs="+", type=int)
    parser.add_argument("--buckets", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument(
        "--simulate-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include a device-to-device copy as a synthetic engine load stage",
    )
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, backend: str) -> None:
    if args.buckets < 1 or args.warmup < 0 or args.iters < 1:
        raise ValueError("buckets and iters must be positive; warmup must be non-negative")
    if any(value < 1 for value in args.max_inflight_buckets):
        raise ValueError("max-inflight-buckets values must be positive")
    if args.max_inflight_engine_groups < 1:
        raise ValueError("max-inflight-engine-groups must be positive")
    if any(value < 0 for value in args.phase_stride_us):
        raise ValueError("phase-stride-us values must be non-negative")
    if "nccl_broadcast" in args.transports and backend != "nccl":
        raise ValueError("nccl_broadcast requires the NCCL backend")


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = _resolve_backend(args.backend)
    _validate_args(args, backend)
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("the NCCL backend requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    engine_groups = build_engine_groups(
        world_size,
        engine_group_size=args.engine_group_size,
        engine_group_sizes=args.engine_group_sizes,
    )
    engine_groups_by_id = {group.engine_id: group for group in engine_groups}
    local_engine_group = _group_for_rank(engine_groups, rank)
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=args.timeout_s))
    control_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    process_groups = _create_engine_process_groups(engine_groups, backend)
    control_process_groups = _create_engine_process_groups(engine_groups, "gloo")
    streams = {
        group.engine_id: (
            torch.cuda.Stream(device=device) if device.type == "cuda" and (rank == 0 or rank in group.ranks) else None
        )
        for group in engine_groups
    }
    local_rank_metadata = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    rank_metadata: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(rank_metadata, local_rank_metadata, group=control_group)
    configs = _build_experiment_configs(args, len(engine_groups))
    all_results = []
    all_trace = []
    try:
        for config_index, config in enumerate(configs):
            tasks = build_transfer_tasks(
                args.buckets,
                engine_groups,
                config.message_bytes,
                transport=config.transport,
            )
            waves = plan_transfer_waves(
                tasks,
                policy=config.engine_wave_policy,
                max_inflight_buckets=config.max_inflight_buckets,
                max_inflight_bytes=config.max_inflight_bytes,
                max_inflight_engine_groups=config.max_inflight_engine_groups,
            )
            trainer_tensors, receiver_tensors, load_tensors = _prepare_tensors(
                bucket_count=args.buckets,
                message_bytes=config.message_bytes,
                rank=rank,
                local_engine_group=local_engine_group,
                device=device,
                simulate_load=args.simulate_load,
            )
            for warmup_iteration in range(args.warmup):
                run_iteration(
                    config,
                    waves,
                    rank=rank,
                    local_engine_group=local_engine_group,
                    engine_groups_by_id=engine_groups_by_id,
                    process_groups=process_groups,
                    control_process_groups=control_process_groups,
                    streams=streams,
                    control_group=control_group,
                    device=device,
                    trainer_tensors=trainer_tensors,
                    receiver_tensors=receiver_tensors,
                    load_tensors=load_tensors,
                    iteration=-(warmup_iteration + 1),
                    simulate_load=args.simulate_load,
                    validate=warmup_iteration == args.warmup - 1,
                )

            local_iteration_ms = []
            local_records = []
            for iteration in range(args.iters):
                elapsed_ms, records = run_iteration(
                    config,
                    waves,
                    rank=rank,
                    local_engine_group=local_engine_group,
                    engine_groups_by_id=engine_groups_by_id,
                    process_groups=process_groups,
                    control_process_groups=control_process_groups,
                    streams=streams,
                    control_group=control_group,
                    device=device,
                    trainer_tensors=trainer_tensors,
                    receiver_tensors=receiver_tensors,
                    load_tensors=load_tensors,
                    iteration=iteration,
                    simulate_load=args.simulate_load,
                    validate=False,
                )
                local_iteration_ms.append(elapsed_ms)
                local_records.extend(records)

            payload = {
                "rank": rank,
                "hostname": socket.gethostname(),
                "iteration_ms": local_iteration_ms,
                "records": local_records,
            }
            gathered: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(gathered, payload, group=control_group)
            if rank == 0:
                complete_gathered = [item for item in gathered if item is not None]
                result = _aggregate_experiment(config, waves, complete_gathered, engine_groups=engine_groups)
                result["execution_order"] = config_index
                all_results.append(result)
                if args.include_trace:
                    all_trace.extend(
                        {
                            "experiment": asdict(config),
                            **record,
                        }
                        for item in complete_gathered
                        for record in item["records"]
                    )
                print(
                    f"[{config_index + 1}/{len(configs)}] {config.transport} "
                    f"{config.engine_wave_policy} bytes={config.message_bytes} "
                    f"buckets={config.max_inflight_buckets} stride={config.phase_stride_us}us: "
                    f"p50={result['weight_sync_total_ms']['p50']:.3f} ms"
                )

        if rank == 0:
            report = {
                "schema_version": 2,
                "framework": "slime",
                "run_id": f"{socket.gethostname()}-{time.time_ns()}",
                "world_size": world_size,
                "trainer_rank": 0,
                "backend": backend,
                "device_type": device.type,
                "communicator_layout": "one trainer-plus-engine process group per engine",
                "ranks": [item for item in rank_metadata if item is not None],
                "engine_groups": [{**asdict(group), "ranks": list(group.ranks)} for group in engine_groups],
                "bucket_count": args.buckets,
                "warmup": args.warmup,
                "iters": args.iters,
                "simulate_load": args.simulate_load,
                "seed": args.seed,
                "kernel_observed": False,
                "gpu_timestamp_semantics": "event-bracket" if device.type == "cuda" else None,
                "timestamp_domain": (
                    "single-node-process-monotonic"
                    if len({item["hostname"] for item in rank_metadata if item is not None}) == 1
                    else "host-local-process-monotonic"
                ),
                "clock_sync_error_bound_us": None,
                "automatic_policy_eligible": False,
                "automatic_policy_ineligibility_reasons": [
                    "transfer timing brackets framework operations rather than observing exact kernels",
                    "no measured cross-rank clock-synchronization error bound",
                ],
                "timing_sources": {
                    "transfer_duration": ("cuda_event_bracket" if device.type == "cuda" else "host_call_bracket"),
                    "weight_sync_total": "maximum per-rank host duration after a shared Gloo start barrier",
                    "rank_skew": ("single-node process monotonic clock; omitted across hosts"),
                    "engine_load": "synthetic device-to-device copy",
                    "idle": "total duration minus summed per-rank transfer/load durations; overlap is clamped at zero",
                    "launch_alignment": "one Gloo barrier per trainer-to-engine transfer; every policy executes the same count",
                },
                "results": all_results,
            }
            if args.include_trace:
                report["trace"] = all_trace
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {args.output_json}")
    finally:
        for process_group in reversed(list(control_process_groups.values())):
            if process_group is not None:
                dist.destroy_process_group(process_group)
        for process_group in reversed(list(process_groups.values())):
            if process_group is not None:
                dist.destroy_process_group(process_group)
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
