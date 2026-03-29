from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from slime.utils.types import Sample

TRACE_VERSION = 1
SGLANG_TRACE_META_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "pd_prefill_bootstrap_queue_duration",
    "pd_prefill_forward_duration",
    "pd_prefill_transfer_queue_duration",
    "pd_prefill_retry_count",
    "pd_decode_prealloc_duration",
    "pd_decode_transfer_duration",
    "pd_decode_forward_duration",
    "pd_bootstrap_duration",
    "pd_alloc_waiting_duration",
    "pd_transfer_speed_gb_s",
    "pd_transfer_total_mb",
)

logger = logging.getLogger(__name__)
_TRACE_STACK: contextvars.ContextVar[tuple[tuple[str, str], ...]] = contextvars.ContextVar(
    "slime_trace_stack",
    default=(),
)
_TRACE_HANDLE_STACK: contextvars.ContextVar[tuple[tuple[TraceHandle, ...], ...]] = contextvars.ContextVar(
    "slime_trace_handle_stack",
    default=(),
)
_TRACE_AUTO_INFER_WARNED: set[str] = set()


@dataclass
class TraceHandle:
    trace_id: str
    carrier: dict[str, Any]
    sample_id: int | str | None = None
    group_id: int | str | None = None
    attempt: int = 0
    parent_span_id: str | None = None


@dataclass
class TraceSpanContext:
    target: Sample | TraceHandle | list[Sample | TraceHandle]
    handles: list[TraceHandle]
    end_attrs: dict[str, Any] = field(default_factory=dict)
    end_events: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    def set(self, key: str, value: Any) -> TraceSpanContext:
        self.end_attrs[key] = value
        self._sync_end_events({key: value})
        return self

    def update(self, attrs: dict[str, Any] | None) -> TraceSpanContext:
        if attrs:
            self.end_attrs.update(attrs)
            self._sync_end_events(attrs)
        return self

    def set_attr(self, key: str, value: Any) -> TraceSpanContext:
        return self.set(key, value)

    def update_attrs(self, attrs: dict[str, Any] | None) -> TraceSpanContext:
        return self.update(attrs)

    def build_end_attrs(self) -> dict[str, Any] | None:
        return dict(self.end_attrs) or None

    def finalize(self, end_events: list[dict[str, Any]]) -> None:
        self.end_events = end_events
        self.closed = True
        if self.end_attrs:
            self._sync_end_events(self.end_attrs)

    def _sync_end_events(self, attrs: dict[str, Any]) -> None:
        if not self.end_events or not attrs:
            return
        for event in self.end_events:
            event.setdefault("attrs", {})
            event["attrs"].update(attrs)


def _noop_handle() -> TraceHandle:
    return TraceHandle(
        trace_id="",
        carrier={
            "version": TRACE_VERSION,
            "trace_id": "",
            "events": [],
            "sample_id": None,
            "group_id": None,
            "attempt": 0,
        },
    )


def _log_trace_error(action: str, exc: Exception) -> None:
    logger.debug("trace %s skipped: %s", action, exc, exc_info=True)


def _new_trace_id() -> str:
    return uuid.uuid4().hex


def _new_span_id() -> str:
    return uuid.uuid4().hex


def build_sglang_meta_trace_attrs(meta: dict[str, Any]) -> dict[str, Any]:
    attrs = {key: meta[key] for key in SGLANG_TRACE_META_KEYS if key in meta and meta[key] is not None}
    attrs["finish_reason"] = meta["finish_reason"]["type"]
    return attrs


def _ensure_trace_carrier(
    carrier: dict[str, Any] | None,
    *,
    trace_id: str | None = None,
    sample_id: int | str | None = None,
    group_id: int | str | None = None,
    attempt: int = 0,
) -> dict[str, Any]:
    if carrier is None:
        carrier = {}
    carrier.setdefault("version", TRACE_VERSION)
    carrier.setdefault("trace_id", trace_id or _new_trace_id())
    carrier.setdefault("events", [])
    if sample_id is not None:
        carrier["sample_id"] = sample_id
    else:
        carrier.setdefault("sample_id", None)
    if group_id is not None:
        carrier["group_id"] = group_id
    else:
        carrier.setdefault("group_id", None)
    carrier["attempt"] = int(carrier.get("attempt", attempt))
    return carrier


def bind_trace(sample: Sample) -> TraceHandle:
    try:
        sample.trace = _ensure_trace_carrier(
            getattr(sample, "trace", None),
            sample_id=sample.index,
            group_id=sample.group_index,
        )
        return TraceHandle(
            trace_id=sample.trace["trace_id"],
            carrier=sample.trace,
            sample_id=sample.trace.get("sample_id"),
            group_id=sample.trace.get("group_id"),
            attempt=int(sample.trace.get("attempt", 0)),
        )
    except Exception as exc:
        _log_trace_error("bind", exc)
        return _noop_handle()


def bind_trace_carrier(
    carrier: dict[str, Any] | None,
    *,
    trace_id: str | None = None,
    sample_id: int | str | None = None,
    group_id: int | str | None = None,
    attempt: int = 0,
    parent_span_id: str | None = None,
) -> TraceHandle:
    try:
        trace = _ensure_trace_carrier(
            carrier,
            trace_id=trace_id,
            sample_id=sample_id,
            group_id=group_id,
            attempt=attempt,
        )
        return TraceHandle(
            trace_id=trace["trace_id"],
            carrier=trace,
            sample_id=trace.get("sample_id"),
            group_id=trace.get("group_id"),
            attempt=int(trace.get("attempt", 0)),
            parent_span_id=parent_span_id,
        )
    except Exception as exc:
        _log_trace_error("bind_carrier", exc)
        handle = _noop_handle()
        handle.parent_span_id = parent_span_id
        return handle


def export_trace(handle: TraceHandle) -> dict[str, Any]:
    try:
        return {
            "version": TRACE_VERSION,
            "trace_id": handle.trace_id,
            "sample_id": handle.sample_id,
            "group_id": handle.group_id,
            "attempt": handle.attempt,
            "parent_span_id": handle.parent_span_id or _get_current_parent_span_id(handle.trace_id),
        }
    except Exception as exc:
        _log_trace_error("export", exc)
        return {
            "version": TRACE_VERSION,
            "trace_id": "",
            "sample_id": None,
            "group_id": None,
            "attempt": 0,
            "parent_span_id": None,
        }


def import_trace(payload: dict[str, Any], carrier: dict[str, Any] | None = None) -> TraceHandle:
    try:
        return bind_trace_carrier(
            carrier,
            trace_id=payload.get("trace_id"),
            sample_id=payload.get("sample_id"),
            group_id=payload.get("group_id"),
            attempt=int(payload.get("attempt", 0)),
            parent_span_id=payload.get("parent_span_id"),
        )
    except Exception as exc:
        _log_trace_error("import", exc)
        return _noop_handle()


def trace_event(
    target: Sample | TraceHandle | list[Sample | TraceHandle], name: str, *, attrs: dict[str, Any] | None = None
):
    try:
        timestamp = time.time()
        for handle in _coerce_handles(target):
            _append_event(handle, kind="event", name=name, timestamp=timestamp, attrs=attrs)
    except Exception as exc:
        _log_trace_error(f"event:{name}", exc)


@contextmanager
def trace_span(
    target: Sample | TraceHandle | list[Sample | TraceHandle],
    name: str,
    *,
    attrs: dict[str, Any] | None = None,
    record_error: bool = True,
):
    try:
        handles = _coerce_handles(target)
    except Exception as exc:
        _log_trace_error(f"span:{name}", exc)
        handles = []

    if not handles:
        yield target
        return

    timestamp = time.time()
    stack_before = _TRACE_STACK.get()
    handle_stack_before = _TRACE_HANDLE_STACK.get()
    span_records: list[tuple[TraceHandle, str]] = []
    new_entries: list[tuple[str, str]] = []

    try:
        for handle in handles:
            span_id = _new_span_id()
            parent_span_id = handle.parent_span_id or _get_current_parent_span_id(handle.trace_id, stack=stack_before)
            _append_event(
                handle,
                kind="span_start",
                name=name,
                timestamp=timestamp,
                span_id=span_id,
                parent_span_id=parent_span_id,
                attrs=attrs,
            )
            span_records.append((handle, span_id))
            new_entries.append((handle.trace_id, span_id))
        token = _TRACE_STACK.set(stack_before + tuple(new_entries))
        handle_token = _TRACE_HANDLE_STACK.set(handle_stack_before + (tuple(handles),))
    except Exception as exc:
        _log_trace_error(f"span:{name}", exc)
        yield target
        return

    span_context = TraceSpanContext(
        target=handles[0] if len(handles) == 1 else handles,
        handles=handles,
    )

    try:
        yield span_context
    except Exception as exc:
        try:
            end_attrs = span_context.build_end_attrs()
            if record_error:
                error_attrs = {"error_type": type(exc).__name__, "error_message": str(exc)}
                if end_attrs:
                    end_attrs.update(error_attrs)
                else:
                    end_attrs = error_attrs
            span_context.finalize(_record_span_end(span_records, name=name, attrs=end_attrs))
        except Exception as trace_exc:
            _log_trace_error(f"span_end:{name}", trace_exc)
        raise
    else:
        try:
            span_context.finalize(_record_span_end(span_records, name=name, attrs=span_context.build_end_attrs()))
        except Exception as exc:
            _log_trace_error(f"span_end:{name}", exc)
    finally:
        try:
            _TRACE_STACK.reset(token)
        except Exception as exc:
            _log_trace_error(f"span_reset:{name}", exc)
        try:
            _TRACE_HANDLE_STACK.reset(handle_token)
        except Exception as exc:
            _log_trace_error(f"span_handle_reset:{name}", exc)


def trace_next_attempt(
    target: Sample | TraceHandle | list[Sample | TraceHandle],
    *,
    attrs: dict[str, Any] | None = None,
):
    try:
        handles = _coerce_handles(target)
        for handle in handles:
            next_attempt = int(handle.carrier.get("attempt", 0)) + 1
            handle.carrier["attempt"] = next_attempt
            handle.attempt = next_attempt
            attempt_attrs = {"attempt": next_attempt}
            if attrs:
                attempt_attrs.update(attrs)
            trace_event(handle, "attempt_start", attrs=attempt_attrs)
        if not handles:
            return target
        return handles[0] if len(handles) == 1 else handles
    except Exception as exc:
        _log_trace_error("next_attempt", exc)
        return target


def trace_function(
    name: str,
    *,
    target: str | None = None,
    target_getter: Callable[..., Sample | TraceHandle | list[Sample | TraceHandle] | None] | None = None,
    attrs_getter: Callable[..., dict[str, Any] | None] | None = None,
    record_error: bool = True,
):
    def decorator(fn):
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                resolved_target = _resolve_trace_function_target(
                    fn,
                    args,
                    kwargs,
                    target=target,
                    target_getter=target_getter,
                )
                if resolved_target is None:
                    return await fn(*args, **kwargs)
                attrs = _resolve_trace_function_attrs(fn, args, kwargs, attrs_getter=attrs_getter)
                with trace_span(resolved_target, name, attrs=attrs, record_error=record_error):
                    return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            resolved_target = _resolve_trace_function_target(
                fn,
                args,
                kwargs,
                target=target,
                target_getter=target_getter,
            )
            if resolved_target is None:
                return fn(*args, **kwargs)
            attrs = _resolve_trace_function_attrs(fn, args, kwargs, attrs_getter=attrs_getter)
            with trace_span(resolved_target, name, attrs=attrs, record_error=record_error):
                return fn(*args, **kwargs)

        return sync_wrapper

    return decorator


def _record_span_end(
    span_records: list[tuple[TraceHandle, str]],
    *,
    name: str,
    attrs: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    timestamp = time.time()
    events = []
    for handle, span_id in span_records:
        events.append(
            _append_event(
                handle,
                kind="span_end",
                name=name,
                timestamp=timestamp,
                span_id=span_id,
                attrs=attrs,
            )
        )
    return events


def _append_event(
    handle: TraceHandle,
    *,
    kind: str,
    name: str,
    timestamp: float,
    attrs: dict[str, Any] | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
) -> dict[str, Any]:
    event = {
        "type": kind,
        "name": name,
        "ts": timestamp,
        "trace_id": handle.trace_id,
        "sample_id": handle.sample_id,
        "group_id": handle.group_id,
        "attempt": int(handle.carrier.get("attempt", handle.attempt)),
    }
    if span_id is not None:
        event["span_id"] = span_id
    if parent_span_id is not None:
        event["parent_span_id"] = parent_span_id
    if attrs:
        event["attrs"] = dict(attrs)
    handle.carrier["events"].append(event)
    return event


def _coerce_handles(target: Sample | TraceHandle | list[Sample | TraceHandle]) -> list[TraceHandle]:
    target = _adapt_trace_target(target)
    if isinstance(target, TraceHandle):
        return [target]
    if isinstance(target, Sample):
        return [bind_trace(target)]
    if isinstance(target, list):
        handles = []
        for item in target:
            handles.extend(_coerce_handles(item))
        return handles
    return []


def _get_current_parent_span_id(
    trace_id: str,
    *,
    stack: tuple[tuple[str, str], ...] | None = None,
) -> str | None:
    stack = _TRACE_STACK.get() if stack is None else stack
    for current_trace_id, span_id in reversed(stack):
        if current_trace_id == trace_id:
            return span_id
    return None


def _resolve_trace_function_target(
    fn,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    target: str | None,
    target_getter: Callable[..., Sample | TraceHandle | list[Sample | TraceHandle] | None] | None,
):
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    except Exception as exc:
        _log_trace_error(f"trace_function_bind:{getattr(fn, '__qualname__', fn)}", exc)
        bound = None

    if target is not None:
        if bound is None or target not in bound.arguments:
            logger.warning(
                "trace_function target '%s' not found for %s; tracing disabled for this call",
                target,
                getattr(fn, "__qualname__", repr(fn)),
            )
            return None
        resolved = _normalize_trace_target(bound.arguments.get(target))
        if resolved is None:
            logger.warning(
                "trace_function target '%s' for %s is not a supported trace target; tracing disabled for this call",
                target,
                getattr(fn, "__qualname__", repr(fn)),
            )
        return resolved

    if target_getter is not None:
        try:
            resolved = _normalize_trace_target(target_getter(*args, **kwargs))
            return resolved
        except Exception as exc:
            _log_trace_error(f"trace_function_target_getter:{getattr(fn, '__qualname__', fn)}", exc)
            return None

    inferred = _infer_trace_target(bound.arguments.values() if bound is not None else args)
    if inferred is not None:
        warn_key = getattr(fn, "__module__", "") + "." + getattr(fn, "__qualname__", repr(fn))
        if warn_key not in _TRACE_AUTO_INFER_WARNED:
            _TRACE_AUTO_INFER_WARNED.add(warn_key)
            logger.warning(
                "trace_function auto-inferred target for %s; inference may be ambiguous, prefer explicit target=...",
                getattr(fn, "__qualname__", repr(fn)),
            )
        return inferred

    return _get_current_trace_target()


def _resolve_trace_function_attrs(
    fn,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    attrs_getter: Callable[..., dict[str, Any] | None] | None,
) -> dict[str, Any] | None:
    if attrs_getter is None:
        return None
    try:
        attrs = attrs_getter(*args, **kwargs)
        if attrs is None:
            return None
        if isinstance(attrs, dict):
            return attrs
        logger.warning(
            "trace_function attrs_getter for %s returned non-dict %s; ignoring attrs",
            getattr(fn, "__qualname__", repr(fn)),
            type(attrs).__name__,
        )
        return None
    except Exception as exc:
        _log_trace_error(f"trace_function_attrs_getter:{getattr(fn, '__qualname__', fn)}", exc)
        return None


def _infer_trace_target(values) -> Sample | TraceHandle | list[Sample | TraceHandle] | None:
    for value in values:
        normalized = _normalize_trace_target(value)
        if normalized is not None:
            return normalized
    return None


def _normalize_trace_target(value):
    value = _adapt_trace_target(value)
    if isinstance(value, (Sample, TraceHandle)):
        return value
    if isinstance(value, list) and value:
        if all(_normalize_trace_target(item) is not None for item in value):
            return value
    return None


def _adapt_trace_target(value):
    if value is None:
        return None
    if isinstance(value, (Sample, TraceHandle)):
        return value
    if isinstance(value, list):
        return [_adapt_trace_target(item) for item in value]
    if _looks_like_sample_box(value):
        generation = getattr(value, "generation", None)
        if generation:
            return generation
        return getattr(value, "prompt_sample", None)
    return value


def _get_current_trace_target() -> TraceHandle | list[TraceHandle] | None:
    handle_stack = _TRACE_HANDLE_STACK.get()
    if not handle_stack:
        return None
    current_handles = list(handle_stack[-1])
    if not current_handles:
        return None
    if len(current_handles) == 1:
        return current_handles[0]
    return current_handles


def _looks_like_sample_box(value: Any) -> bool:
    cls = getattr(value, "__class__", None)
    if cls is None or getattr(cls, "__name__", "") != "SampleBox":
        return False
    return hasattr(value, "prompt_sample") and hasattr(value, "generation")
