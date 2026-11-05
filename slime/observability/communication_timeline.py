from __future__ import annotations

import atexit
import contextvars
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

COMMUNICATION_TIMELINE_VERSION = 2
COMMUNICATION_TIMELINE_ENV = "SLIME_COMMUNICATION_TIMELINE"
COMMUNICATION_TIMELINE_RUN_ID_ENV = "SLIME_COMMUNICATION_TIMELINE_RUN_ID"

_STANDARD_FIELDS = (
    "global_step",
    "rollout_id",
    "weight_version",
    "bucket_id",
    "trainer_rank",
    "engine_id",
    "message_bytes",
    "transport",
)
_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "slime_communication_timeline_context",
    default=None,
)
_GLOBAL_LOCK = threading.RLock()
_GLOBAL_TIMELINE: CommunicationTimeline | None = None
_GLOBAL_CONFIGURED = False


class _DisabledCommunicationPhase:
    """Shared no-op phase used after tracing is configured off."""

    enabled = False

    def __enter__(self) -> _DisabledCommunicationPhase:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def update(self, **fields: Any) -> _DisabledCommunicationPhase:
        return self

    def mark_consumer(self) -> _DisabledCommunicationPhase:
        return self

    def mark_api_return(self) -> _DisabledCommunicationPhase:
        return self

    def cancel(self) -> None:
        return None


_DISABLED_PHASE = _DisabledCommunicationPhase()


def _optional_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def _detect_rank() -> int:
    torch = _optional_torch()
    if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def _detect_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _detect_world_size() -> int:
    torch = _optional_torch()
    if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def _resolve_path(template: str, *, rank: int, local_rank: int, role: str, world_size: int) -> Path:
    values = {
        "rank": rank,
        "trainer_rank": rank,
        "local_rank": local_rank,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "role": role,
        "world_size": world_size,
    }
    try:
        resolved = template.format_map(values)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "communication timeline path supports only {rank}, {trainer_rank}, {local_rank}, "
            "{pid}, {hostname}, {role}, and {world_size} placeholders"
        ) from exc
    path = Path(resolved).expanduser()
    if role != "trainer" and "{role}" not in template and "{pid}" not in template:
        suffix = path.suffix
        stem = path.name[: -len(suffix)] if suffix else path.name
        path = path.with_name(f"{stem}.role-{role}{suffix}")
    if world_size > 1 and not any(f"{{{name}}}" in template for name in ("rank", "trainer_rank", "pid")):
        suffix = path.suffix
        stem = path.name[: -len(suffix)] if suffix else path.name
        path = path.with_name(f"{stem}.rank-{rank}{suffix}")
    return path


def _message_bytes(named_tensors) -> int:
    total = 0
    for item in named_tensors:
        tensor = item[1] if isinstance(item, tuple) else item
        total += int(tensor.numel()) * int(tensor.element_size())
    return total


@dataclass
class _PendingRecord:
    record: dict[str, Any] | None
    gpu_start: Any = None
    gpu_end: Any = None


class CommunicationTimeline:
    """Append-only trainer communication timeline.

    CUDA events are queried without synchronizing the training stream during normal
    operation. ``close()`` waits only for the outstanding end events so the final
    records are not lost at process shutdown.
    """

    def __init__(
        self,
        path: str,
        *,
        rank: int | None = None,
        local_rank: int | None = None,
        world_size: int | None = None,
        role: str = "trainer",
        run_id: str | None = None,
    ) -> None:
        self.rank = _detect_rank() if rank is None else rank
        self.local_rank = _detect_local_rank() if local_rank is None else local_rank
        self.world_size = _detect_world_size() if world_size is None else world_size
        self.role = role
        self.run_id = run_id or os.environ.get(COMMUNICATION_TIMELINE_RUN_ID_ENV) or uuid.uuid4().hex
        self.path = _resolve_path(
            path,
            rank=self.rank,
            local_rank=self.local_rank,
            role=role,
            world_size=self.world_size,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()
        self._pending: dict[int, _PendingRecord] = {}
        self._sequence_id = 0
        self._next_write_sequence_id = 0
        self._closed = False
        self._write_failed = False

    def next_sequence_id(self) -> int:
        with self._lock:
            sequence_id = self._sequence_id
            self._sequence_id += 1
            return sequence_id

    def new_span(self, operation: str, fields: Mapping[str, Any]) -> CommunicationPhase:
        return CommunicationPhase(self, operation, dict(fields))

    def emit_event(self, operation: str, fields: Mapping[str, Any]) -> None:
        now_wall = time.time_ns()
        now_monotonic = time.perf_counter_ns()
        record = self._base_record(operation, fields)
        record.update(
            {
                "record_type": "event",
                "api_launch_timestamp_ns": now_wall,
                "api_return_timestamp_ns": now_wall,
                "completion_timestamp_ns": now_wall,
                "consumer_timestamp_ns": fields.get("consumer_timestamp_ns"),
                "api_launch_monotonic_ns": now_monotonic,
                "completion_monotonic_ns": now_monotonic,
                "duration_ns": 0,
                "gpu_start_timestamp_ns": None,
                "gpu_end_timestamp_ns": None,
                "gpu_elapsed_ns": None,
                "status": "ok",
            }
        )
        self._queue(_PendingRecord(record=record))
        self.flush(block=False)

    def _base_record(self, operation: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        context = dict(_CONTEXT.get() or {})
        context.update(fields)
        standard = {name: context.pop(name, None) for name in _STANDARD_FIELDS}
        if standard["trainer_rank"] is None:
            standard["trainer_rank"] = self.rank
        sequence_id = self.next_sequence_id()
        logical_parts = (
            standard["global_step"],
            standard["rollout_id"],
            standard["weight_version"],
            standard["bucket_id"],
            operation,
        )
        logical_operation_id = "/".join("-" if value is None else str(value) for value in logical_parts)
        if "wave_id" in context:
            logical_operation_id += f"/wave/{context['wave_id']}"
        return {
            "schema_version": COMMUNICATION_TIMELINE_VERSION,
            "framework": "slime",
            "gpu_timestamp_semantics": "event-bracket",
            "timestamp_domain": "process-realtime-projected-cuda-event",
            "clock_sync_error_bound_us": None,
            "run_id": self.run_id,
            "role": self.role,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "sequence_id": sequence_id,
            "logical_operation_id": logical_operation_id,
            "operation": operation,
            **standard,
            "metadata": context,
        }

    def finish_span(self, span: CommunicationPhase, error: BaseException | None) -> None:
        if span.cancelled:
            self._queue(_PendingRecord(record=None), sequence_id=span.record["sequence_id"])
            self.flush(block=False)
            return
        end_wall = time.time_ns()
        end_monotonic = time.perf_counter_ns()
        if span.gpu_end is not None:
            try:
                span.gpu_end.record()
            except Exception as exc:
                span.record["metadata"]["gpu_event_error"] = str(exc)
                span.gpu_start = span.gpu_end = None
        record = span.record
        record.update(
            {
                "api_return_timestamp_ns": span.api_return_timestamp_ns or end_wall,
                "completion_timestamp_ns": end_wall,
                "consumer_timestamp_ns": span.consumer_timestamp_ns,
                "completion_monotonic_ns": end_monotonic,
                "duration_ns": end_monotonic - span.start_monotonic_ns,
                "gpu_start_timestamp_ns": None,
                "gpu_end_timestamp_ns": None,
                "gpu_elapsed_ns": None,
                "status": "error" if error is not None else "ok",
            }
        )
        if error is not None:
            record["metadata"]["error_type"] = type(error).__name__
            record["metadata"]["error_message"] = str(error)
        pending = _PendingRecord(record=record, gpu_start=span.gpu_start, gpu_end=span.gpu_end)
        self._queue(pending)
        self.flush(block=False)

    def _queue(self, pending: _PendingRecord, *, sequence_id: int | None = None) -> None:
        if sequence_id is None:
            assert pending.record is not None
            sequence_id = pending.record["sequence_id"]
        with self._lock:
            self._pending[sequence_id] = pending

    def flush(self, *, block: bool = False) -> None:
        with self._lock:
            while self._next_write_sequence_id in self._pending:
                pending = self._pending[self._next_write_sequence_id]
                if pending.record is None:
                    del self._pending[self._next_write_sequence_id]
                    self._next_write_sequence_id += 1
                    continue
                if pending.gpu_end is not None:
                    try:
                        if block:
                            pending.gpu_end.synchronize()
                        elif not pending.gpu_end.query():
                            break
                        elapsed_ns = int(round(pending.gpu_start.elapsed_time(pending.gpu_end) * 1_000_000))
                        gpu_start_ns = pending.record["api_launch_timestamp_ns"]
                        pending.record["gpu_start_timestamp_ns"] = gpu_start_ns
                        pending.record["gpu_end_timestamp_ns"] = gpu_start_ns + elapsed_ns
                        pending.record["gpu_elapsed_ns"] = elapsed_ns
                    except Exception as exc:
                        pending.record["metadata"]["gpu_timing_error"] = str(exc)
                self._write_unlocked(pending.record)
                del self._pending[self._next_write_sequence_id]
                self._next_write_sequence_id += 1
            try:
                self._file.flush()
            except OSError as exc:
                self._mark_write_failed(exc)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
        try:
            self.flush(block=True)
        finally:
            with self._lock:
                if not self._closed:
                    try:
                        self._file.close()
                    except OSError as exc:
                        self._mark_write_failed(exc)
                    self._closed = True

    def _write_unlocked(self, record: Mapping[str, Any]) -> None:
        if self._write_failed:
            return
        try:
            self._file.write(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n")
        except OSError as exc:
            self._mark_write_failed(exc)

    def _mark_write_failed(self, exc: OSError) -> None:
        if not self._write_failed:
            logger.warning("communication timeline disabled after write failure: %s", exc)
            self._write_failed = True


@dataclass
class CommunicationPhase:
    enabled: ClassVar[bool] = True
    timeline: CommunicationTimeline | None
    operation: str
    fields: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict, init=False)
    start_monotonic_ns: int = field(default=0, init=False)
    consumer_timestamp_ns: int | None = field(default=None, init=False)
    api_return_timestamp_ns: int | None = field(default=None, init=False)
    gpu_start: Any = field(default=None, init=False)
    gpu_end: Any = field(default=None, init=False)
    cancelled: bool = field(default=False, init=False)
    _nvtx: bool = field(default=False, init=False)

    def __enter__(self) -> CommunicationPhase:
        if self.timeline is None:
            return self
        start_wall = time.time_ns()
        self.start_monotonic_ns = time.perf_counter_ns()
        self.record = self.timeline._base_record(self.operation, self.fields)
        self.record.update(
            {
                "record_type": "span",
                "api_launch_timestamp_ns": start_wall,
                "api_launch_monotonic_ns": self.start_monotonic_ns,
            }
        )
        torch = _optional_torch()
        if torch is not None and torch.cuda.is_available():
            try:
                self.record["stream_id"] = int(torch.cuda.current_stream().cuda_stream)
                self.gpu_start = torch.cuda.Event(enable_timing=True)
                self.gpu_end = torch.cuda.Event(enable_timing=True)
                self.gpu_start.record()
                torch.cuda.nvtx.range_push(f"slime.comm/{self.operation}")
                self._nvtx = True
            except Exception as exc:
                self.record["metadata"]["gpu_event_error"] = str(exc)
                self.gpu_start = self.gpu_end = None
        else:
            self.record["stream_id"] = None
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.timeline is None:
            return False
        self._close_nvtx()
        try:
            self.timeline.finish_span(self, exc)
        except Exception as timeline_exc:
            logger.warning("communication timeline phase %s could not be recorded: %s", self.operation, timeline_exc)
        return False

    def _close_nvtx(self) -> None:
        if self._nvtx:
            torch = _optional_torch()
            try:
                torch.cuda.nvtx.range_pop()
            except Exception as nvtx_exc:
                self.record["metadata"]["nvtx_error"] = str(nvtx_exc)
            self._nvtx = False

    def mark_api_return(self) -> CommunicationPhase:
        """End the API/NVTX range while retaining the deferred GPU end event."""
        if self.timeline is not None and self.api_return_timestamp_ns is None:
            self.api_return_timestamp_ns = time.time_ns()
            self._close_nvtx()
        return self

    def update(self, **fields: Any) -> CommunicationPhase:
        if self.timeline is None:
            return self
        for name, value in fields.items():
            if name in _STANDARD_FIELDS:
                self.record[name] = value
            else:
                self.record.setdefault("metadata", {})[name] = value
        return self

    def mark_consumer(self) -> CommunicationPhase:
        self.consumer_timestamp_ns = time.time_ns()
        return self

    def cancel(self) -> None:
        self.cancelled = True


def configure_communication_timeline(
    path: str | None = None,
    *,
    rank: int | None = None,
    local_rank: int | None = None,
    world_size: int | None = None,
    role: str = "trainer",
    run_id: str | None = None,
) -> CommunicationTimeline | None:
    """Configure the process-local timeline; ``None`` uses the environment."""

    global _GLOBAL_CONFIGURED, _GLOBAL_TIMELINE
    path = path or os.environ.get(COMMUNICATION_TIMELINE_ENV)
    with _GLOBAL_LOCK:
        if _GLOBAL_TIMELINE is not None:
            _GLOBAL_TIMELINE.close()
        _GLOBAL_TIMELINE = (
            CommunicationTimeline(
                path,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                role=role,
                run_id=run_id,
            )
            if path
            else None
        )
        _GLOBAL_CONFIGURED = True
        return _GLOBAL_TIMELINE


def get_communication_timeline() -> CommunicationTimeline | None:
    global _GLOBAL_CONFIGURED
    with _GLOBAL_LOCK:
        if not _GLOBAL_CONFIGURED:
            configure_communication_timeline()
        return _GLOBAL_TIMELINE


def _configured_timeline() -> CommunicationTimeline | None:
    if _GLOBAL_CONFIGURED:
        return _GLOBAL_TIMELINE
    return get_communication_timeline()


def communication_phase(operation: str, **fields: Any) -> CommunicationPhase | _DisabledCommunicationPhase:
    timeline = _configured_timeline()
    if timeline is None:
        return _DISABLED_PHASE
    return CommunicationPhase(timeline, operation, fields)


def communication_event(operation: str, **fields: Any) -> None:
    timeline = _configured_timeline()
    if timeline is not None:
        timeline.emit_event(operation, fields)


@contextmanager
def communication_context(**fields: Any) -> Iterator[None]:
    if _configured_timeline() is None:
        yield
        return
    current = dict(_CONTEXT.get() or {})
    current.update(fields)
    token = _CONTEXT.set(current)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def iter_communication_buckets(chunks, *, operation: str = "weight_convert", start_bucket_id: int = 0):
    """Yield chunks while timing the generator work that produces each bucket."""

    iterator = iter(chunks)
    bucket_id = start_bucket_id
    while True:
        with communication_phase(operation, bucket_id=bucket_id) as phase:
            try:
                chunk = next(iterator)
            except StopIteration:
                phase.cancel()
                break
            if phase.enabled:
                phase.update(message_bytes=_message_bytes(chunk))
        yield bucket_id, chunk
        bucket_id += 1


def flush_communication_timeline(*, block: bool = False) -> None:
    timeline = _configured_timeline()
    if timeline is not None:
        timeline.flush(block=block)


def close_communication_timeline() -> None:
    global _GLOBAL_CONFIGURED, _GLOBAL_TIMELINE
    with _GLOBAL_LOCK:
        timeline, _GLOBAL_TIMELINE = _GLOBAL_TIMELINE, None
        _GLOBAL_CONFIGURED = True
    if timeline is not None:
        timeline.close()


atexit.register(close_communication_timeline)
