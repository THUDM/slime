import json
import logging
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any

PARSER_VERSION = "slime.request_time_stats.v1"

_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_RID_KEYS = ("rid", "request_id", "sglang_request_id")
_REQ_TIME_STATS_RE = re.compile(r"ReqTimeStats\((?P<header>.*?)\):\s*(?P<body>.*)$")
_REQ_TIME_STATS_HEADER_RE = re.compile(r"(?P<key>[A-Za-z_#][A-Za-z0-9_# ]*?)=(?P<value>[^,)]+)")
_REQ_TIME_STATS_DURATION_CALL_RE = re.compile(rf"(?P<key>[A-Za-z_#][A-Za-z0-9_# ]*?)\((?P<value>{_NUMBER_RE})ms\)")
_REQ_TIME_STATS_VALUE_RE = re.compile(
    rf"(?P<key>[A-Za-z_#][A-Za-z0-9_# ]*?)\s*=\s*(?P<value>{_NUMBER_RE})\s*(?P<unit>GB/s|MB|ms)?"
)


class ReqTimeStatsLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "ReqTimeStats(" in record.getMessage()


class AppendOnlyFileHandler(logging.Handler):
    terminator = "\n"

    def __init__(self, filename: str, encoding: str = "utf-8"):
        super().__init__()
        filename = filename.format(pid=os.getpid(), hostname=socket.gethostname())
        self.baseFilename = os.path.abspath(os.path.expanduser(filename))
        self.encoding = encoding
        os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)
        self._fd = os.open(self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record) + self.terminator
            os.write(self._fd, message.encode(self.encoding, errors="replace"))
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            if getattr(self, "_fd", None) is not None:
                os.close(self._fd)
                self._fd = None
        finally:
            super().close()


def parse_request_time_stats_line(line: str, source: str | None = None) -> tuple[str, dict[str, Any]] | None:
    match = _REQ_TIME_STATS_RE.search(line)
    if not match:
        return None

    header = {
        _normalize_request_time_stats_key(item.group("key")): item.group("value").strip()
        for item in _REQ_TIME_STATS_HEADER_RE.finditer(match.group("header"))
    }
    rid = header.get("rid")
    if not rid:
        return None

    values: dict[str, float] = {}
    body = match.group("body")
    for item in _REQ_TIME_STATS_DURATION_CALL_RE.finditer(body):
        key = _normalize_request_time_stats_key(item.group("key"))
        values[key] = float(item.group("value")) / 1000.0

    for item in _REQ_TIME_STATS_VALUE_RE.finditer(body):
        key = _normalize_request_time_stats_key(item.group("key"))
        value = float(item.group("value"))
        unit = item.group("unit")
        if unit == "ms":
            value /= 1000.0
        values[key] = value

    request_type = str(header.get("type") or "").lower()
    attrs: dict[str, Any] = {
        "parser_version": PARSER_VERSION,
        "sglang_request_time_stats_type": request_type,
    }
    if source:
        attrs["sglang_request_time_stats_source"] = source

    if request_type == "prefill":
        bootstrap_queue = _request_time_stats_duration(values, "bootstrap_queue_duration")
        if bootstrap_queue is None:
            bootstrap_queue = _request_time_stats_sum(values, "bootstrap_duration", "alloc_wait_duration")
        if bootstrap_queue is not None:
            attrs["pd_prefill_bootstrap_queue_duration"] = bootstrap_queue

        prefill_forward = _request_time_stats_duration(values, "forward_duration")
        if prefill_forward is not None:
            attrs["pd_prefill_forward_duration"] = prefill_forward

        retry_count = values.get("retry_count")
        if retry_count is not None:
            attrs["pd_prefill_retry_count"] = int(retry_count)

        transfer_speed = values.get("transfer_speed")
        if transfer_speed is not None:
            attrs["pd_transfer_speed_gb_s"] = transfer_speed

        transfer_total = values.get("transfer_total")
        if transfer_total is not None:
            attrs["pd_transfer_total_mb"] = transfer_total

    elif request_type == "decode":
        prealloc = _request_time_stats_duration(values, "prealloc_queue_duration", "prealloc_duration")
        if prealloc is None:
            prealloc = _request_time_stats_sum(values, "bootstrap_duration", "alloc_wait_duration")
        if prealloc is not None:
            attrs["pd_decode_prealloc_duration"] = prealloc

        transfer = _request_time_stats_duration(values, "transfer_duration")
        if transfer is not None:
            attrs["pd_decode_transfer_duration"] = transfer

        decode_forward = _request_time_stats_duration(values, "forward_duration")
        if decode_forward is not None:
            attrs["pd_decode_forward_duration"] = decode_forward

    else:
        queue_duration = _request_time_stats_duration(values, "queue_duration")
        if queue_duration is not None:
            attrs["sglang_queue_duration"] = queue_duration
        forward_duration = _request_time_stats_duration(values, "forward_duration")
        if forward_duration is not None:
            attrs["sglang_forward_duration"] = forward_duration

    bootstrap_duration = _request_time_stats_duration(values, "bootstrap_duration")
    if bootstrap_duration is not None:
        attrs["pd_bootstrap_duration"] = bootstrap_duration

    alloc_wait_duration = _request_time_stats_duration(values, "alloc_wait_duration")
    if alloc_wait_duration is not None:
        attrs["pd_alloc_waiting_duration"] = alloc_wait_duration

    for source_key, attr_key in (
        ("input_len", "sglang_input_len"),
        ("cached_input_len", "sglang_cached_input_len"),
        ("output_len", "sglang_output_len"),
        ("bootstrap_room", "sglang_bootstrap_room"),
    ):
        value = header.get(source_key)
        if value is None:
            continue
        try:
            attrs[attr_key] = int(value)
        except ValueError:
            attrs[attr_key] = value

    return str(rid), attrs


def parse_request_time_stats_record(
    record: dict[str, Any], source: str | None = None
) -> tuple[str, dict[str, Any]] | None:
    rid = next((record.get(key) for key in _RID_KEYS if record.get(key) is not None), None)
    if rid is None:
        return None

    attrs: dict[str, Any] = {}
    nested_attrs = record.get("attrs")
    if isinstance(nested_attrs, dict):
        attrs.update({str(key): value for key, value in nested_attrs.items()})

    excluded = set(_RID_KEYS) | {"attrs"}
    attrs.update({str(key): value for key, value in record.items() if key not in excluded})
    if source:
        attrs.setdefault("request_time_stats_source", source)
    return str(rid), attrs


def parse_request_time_stats_record_line(line: str, source: str | None = None) -> tuple[str, dict[str, Any]] | None:
    if not line.lstrip().startswith("{"):
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    return parse_request_time_stats_record(record, source=source)


def iter_request_time_stats_files(path: Path | None, exclude: set[Path] | None = None) -> list[Path]:
    if path is None or not path.exists():
        return []

    exclude = {item.resolve() for item in exclude or set()}
    if path.is_file():
        resolved = path.resolve()
        return [] if resolved in exclude else [path]

    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.resolve() not in exclude and item.suffix.lower() in {".log", ".jsonl", ".txt"}
    )


def load_request_time_stats(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    files = iter_request_time_stats_files(path)
    records_by_rid: dict[str, dict[str, Any]] = {}
    record_count = 0

    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parsed = parse_request_time_stats_record_line(line, source=str(file_path))
                    if parsed is None:
                        parsed = parse_request_time_stats_line(line, source=str(file_path))
                    if parsed is None:
                        continue
                    rid, attrs = parsed
                    records_by_rid.setdefault(rid, {}).update(attrs)
                    record_count += 1
        except OSError as exc:
            print(f"[request_time_stats] skipped request-time-stats file {file_path}: {exc}", file=sys.stderr)

    summary = {
        "path": str(path) if path is not None else None,
        "file_count": len(files),
        "record_count": record_count,
        "rid_count": len(records_by_rid),
    }
    return records_by_rid, summary


def request_time_stats_mtime(path: Path | None) -> float | None:
    mtimes = []
    for file_path in iter_request_time_stats_files(path):
        try:
            mtimes.append(file_path.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes, default=None)


def _normalize_request_time_stats_key(key: str) -> str:
    key = key.strip().lower().replace(" ", "_")
    aliases = {
        "#retries": "retry_count",
        "retries": "retry_count",
        "bootstrap": "bootstrap_duration",
        "alloc_wait": "alloc_wait_duration",
        "alloc_waiting": "alloc_wait_duration",
        "start": "entry_time",
    }
    return aliases.get(key, key)


def _request_time_stats_duration(values: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _request_time_stats_sum(values: dict[str, float], *keys: str) -> float | None:
    parts = []
    for key in keys:
        value = values.get(key)
        if isinstance(value, (int, float)) and value > 0:
            parts.append(float(value))
    if not parts:
        return None
    return sum(parts)
