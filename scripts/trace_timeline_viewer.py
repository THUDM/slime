#!/usr/bin/env python3
"""Build and serve an interactive timeline viewer for rollout trace dumps.

The viewer consumes a rollout debug dump `.pt` file, extracts per-sample trace
events, rebuilds spans and point events, and writes a lightweight JSON cache
plus a self-contained HTML viewer next to the source file.
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import socketserver
import sys
import time
import types
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

import torch


CACHE_VERSION = 1


class _MissingPickleObject:
    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
            return
        self.__dict__["_raw_state"] = state


_MISSING_PICKLE_GLOBALS: set[tuple[str, str]] = set()


def _ensure_dummy_module(module_name: str) -> types.ModuleType:
    module = sys.modules.get(module_name)
    if isinstance(module, types.ModuleType):
        return module

    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = _ensure_dummy_module(parent_name)
        setattr(parent, child_name, module)
    return module


def _make_dummy_pickle_global(module_name: str, name: str) -> type[_MissingPickleObject]:
    module = _ensure_dummy_module(module_name)
    existing = getattr(module, name, None)
    if isinstance(existing, type):
        return existing

    dummy_type = type(name, (_MissingPickleObject,), {"__module__": module_name})
    setattr(module, name, dummy_type)
    _MISSING_PICKLE_GLOBALS.add((module_name, name))
    return dummy_type


class _DummyFallbackUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            return _make_dummy_pickle_global(module, name)


_DUMMY_FALLBACK_PICKLE_MODULE = types.SimpleNamespace(
    __name__="pickle",
    Unpickler=_DummyFallbackUnpickler,
    load=pickle.load,
    loads=pickle.loads,
)


@dataclass
class TimelinePaths:
    pt_path: Path
    cache_path: Path
    html_path: Path


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _compact_text(value: Any, max_len: int = 256) -> Any:
    value = _json_safe(value)
    if not isinstance(value, str):
        return value
    if len(value) <= max_len:
        return value
    return f"{value[:max_len]}...<truncated:{len(value)}>"


def _safe_duration(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, float(end) - float(start))


def _to_sample_dict(sample: Any) -> dict[str, Any]:
    if hasattr(sample, "to_dict"):
        sample = sample.to_dict()
    if isinstance(sample, dict):
        return sample
    result = {}
    for key in (
        "group_index",
        "index",
        "prompt",
        "response",
        "response_length",
        "reward",
        "metadata",
        "source",
        "status",
        "label",
        "trace",
    ):
        if hasattr(sample, key):
            result[key] = getattr(sample, key)
    return result


def _infer_source(sample: dict[str, Any], metadata: dict[str, Any]) -> Any:
    if sample.get("source") not in (None, ""):
        return sample.get("source")
    if metadata.get("source") not in (None, ""):
        return metadata.get("source")
    if metadata.get("source_name") not in (None, ""):
        return metadata.get("source_name")
    for key, value in metadata.items():
        if "source" in str(key).lower() and value not in (None, ""):
            return value
    return None


def _event_timestamp(event: dict[str, Any]) -> float | None:
    ts = event.get("ts")
    if ts is None:
        return None
    try:
        return float(ts)
    except (TypeError, ValueError):
        return None


def _normalize_trace_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    raw_events = trace.get("events") or []
    normalized = []
    active_stack: list[str] = []

    for order, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            continue
        ts = _event_timestamp(raw_event)
        if ts is None:
            continue

        event = {
            "order": order,
            "ts": ts,
            "type": _json_safe(raw_event.get("type")),
            "name": _json_safe(raw_event.get("name")),
            "attempt": int(raw_event.get("attempt", trace.get("attempt", 0)) or 0),
            "sample_id": _json_safe(raw_event.get("sample_id", trace.get("sample_id"))),
            "group_id": _json_safe(raw_event.get("group_id", trace.get("group_id"))),
            "span_id": _json_safe(raw_event.get("span_id")),
            "parent_span_id": _json_safe(raw_event.get("parent_span_id")),
            "attrs": _json_safe(raw_event.get("attrs") or {}),
        }
        event["inferred_parent_span_id"] = active_stack[-1] if active_stack else None
        normalized.append(event)

        if event["type"] == "span_start" and event["span_id"]:
            active_stack.append(event["span_id"])
            continue

        if event["type"] == "span_end" and event["span_id"]:
            for idx in range(len(active_stack) - 1, -1, -1):
                if active_stack[idx] == event["span_id"]:
                    del active_stack[idx]
                    break

    return normalized


def _span_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or "span")


def _span_type(item: dict[str, Any]) -> str:
    if item["type"] == "event":
        return "point_event"
    if item["type"] == "orphan_end":
        return "orphan_end"
    return item["state"]


def _compute_span_depths(spans: list[dict[str, Any]]) -> dict[str, int]:
    span_by_id = {span["span_id"]: span for span in spans if span.get("span_id")}
    cache: dict[str, int] = {}

    def resolve(span_id: str | None, seen: set[str]) -> int:
        if not span_id or span_id not in span_by_id:
            return 0
        if span_id in cache:
            return cache[span_id]
        if span_id in seen:
            cache[span_id] = 0
            return 0
        seen.add(span_id)
        parent_id = span_by_id[span_id].get("parent_span_id")
        depth = 0 if not parent_id or parent_id not in span_by_id else resolve(parent_id, seen) + 1
        cache[span_id] = depth
        return depth

    for span in spans:
        span_id = span.get("span_id")
        if span_id:
            resolve(span_id, set())
    return cache


def _build_items_from_trace(sample: dict[str, Any], sample_idx: int) -> dict[str, Any] | None:
    trace = sample.get("trace")
    if not isinstance(trace, dict):
        return None

    events = _normalize_trace_events(trace)
    if not events:
        return None

    open_starts: dict[str, dict[str, Any]] = {}
    closed_spans: list[dict[str, Any]] = []
    point_events: list[dict[str, Any]] = []
    orphan_ends: list[dict[str, Any]] = []
    all_timestamps: list[float] = []

    for event in events:
        all_timestamps.append(event["ts"])
        event_type = event["type"]

        if event_type == "span_start" and event["span_id"]:
            open_starts[event["span_id"]] = {
                "type": "span",
                "state": "closed_span",
                "name": event["name"],
                "start_ts": event["ts"],
                "end_ts": None,
                "display_end_ts": None,
                "attempt": event["attempt"],
                "span_id": event["span_id"],
                "parent_span_id": event.get("parent_span_id"),
                "start_attrs": event.get("attrs") or {},
                "end_attrs": {},
            }
            continue

        if event_type == "span_end":
            span_id = event.get("span_id")
            start_record = open_starts.pop(span_id, None) if span_id else None
            if start_record is None:
                orphan_ends.append(
                    {
                        "type": "orphan_end",
                        "state": "orphan_end",
                        "name": event["name"],
                        "ts": event["ts"],
                        "attempt": event["attempt"],
                        "span_id": span_id,
                        "parent_span_id": event.get("parent_span_id") or event.get("inferred_parent_span_id"),
                        "attrs": event.get("attrs") or {},
                    }
                )
                continue

            start_record["end_ts"] = event["ts"]
            start_record["display_end_ts"] = event["ts"]
            start_record["end_attrs"] = event.get("attrs") or {}
            closed_spans.append(start_record)
            continue

        point_events.append(
            {
                "type": "event",
                "state": "point_event",
                "name": event["name"],
                "ts": event["ts"],
                "attempt": event["attempt"],
                "span_id": None,
                "parent_span_id": event.get("inferred_parent_span_id"),
                "attrs": event.get("attrs") or {},
            }
        )

    row_end_ts = max(all_timestamps) if all_timestamps else None
    open_spans = list(open_starts.values())
    all_spans = closed_spans + open_spans
    span_depths = _compute_span_depths(all_spans)
    span_by_id = {span["span_id"]: span for span in all_spans if span.get("span_id")}
    sibling_groups: dict[str | None, list[dict[str, Any]]] = {}

    for span in all_spans:
        sibling_groups.setdefault(span.get("parent_span_id"), []).append(span)

    for siblings in sibling_groups.values():
        siblings.sort(key=lambda item: (item["start_ts"], item["end_ts"] or float("inf"), _span_name(item)))

    def nearest_closed_ancestor_end(span: dict[str, Any]) -> float | None:
        current_parent = span.get("parent_span_id")
        while current_parent:
            parent = span_by_id.get(current_parent)
            if parent is None:
                return None
            if parent.get("end_ts") is not None:
                return float(parent["end_ts"])
            current_parent = parent.get("parent_span_id")
        return None

    for span in open_spans:
        candidates: list[tuple[float, str]] = []
        if row_end_ts is not None:
            candidates.append((row_end_ts, "row_end"))

        siblings = sibling_groups.get(span.get("parent_span_id"), [])
        for sibling in siblings:
            if sibling is span:
                continue
            sibling_start = float(sibling["start_ts"])
            if sibling_start > float(span["start_ts"]):
                candidates.append((sibling_start, "next_sibling_start"))
                break

        ancestor_end = nearest_closed_ancestor_end(span)
        if ancestor_end is not None and ancestor_end > float(span["start_ts"]):
            candidates.append((ancestor_end, "ancestor_end"))

        if candidates:
            display_end_ts, clipped_by = min(candidates, key=lambda item: item[0])
            if display_end_ts <= float(span["start_ts"]):
                display_end_ts = float(span["start_ts"])
                clipped_by = "self"
        else:
            display_end_ts = float(span["start_ts"])
            clipped_by = "self"

        span["state"] = "open_span"
        span["display_end_ts"] = display_end_ts
        span.setdefault("end_attrs", {})
        span["end_attrs"]["clipped_by"] = clipped_by

    for span in all_spans:
        span["depth"] = span_depths.get(span.get("span_id") or "", 0)
        span["lane"] = span["depth"]

    for event in point_events:
        parent_span_id = event.get("parent_span_id")
        event["depth"] = span_depths.get(parent_span_id or "", 0)
        event["lane"] = event["depth"]

    for item in orphan_ends:
        parent_span_id = item.get("parent_span_id")
        item["depth"] = span_depths.get(parent_span_id or "", 0)
        item["lane"] = item["depth"]

    def parent_span_name(parent_span_id: str | None) -> str | None:
        if not parent_span_id:
            return None
        parent = span_by_id.get(parent_span_id)
        if not parent:
            return None
        return parent.get("name")

    all_items: list[dict[str, Any]] = []
    for span in all_spans:
        all_items.append(
            {
                "type": "span",
                "state": span["state"],
                "name": span["name"],
                "start_ts": _round_float(span["start_ts"]),
                "end_ts": _round_float(span["end_ts"]),
                "display_end_ts": _round_float(span["display_end_ts"]),
                "attempt": span["attempt"],
                "span_id": span.get("span_id"),
                "parent_span_id": span.get("parent_span_id"),
                "parent_span_name": parent_span_name(span.get("parent_span_id")),
                "lane": span["lane"],
                "depth": span["depth"],
                "attrs": {
                    "start_attrs": _json_safe(span.get("start_attrs") or {}),
                    "end_attrs": _json_safe(span.get("end_attrs") or {}),
                },
            }
        )

    for event in point_events:
        all_items.append(
            {
                "type": "event",
                "state": "point_event",
                "name": event["name"],
                "ts": _round_float(event["ts"]),
                "attempt": event["attempt"],
                "span_id": None,
                "parent_span_id": event.get("parent_span_id"),
                "parent_span_name": parent_span_name(event.get("parent_span_id")),
                "lane": event["lane"],
                "depth": event["depth"],
                "attrs": _json_safe(event.get("attrs") or {}),
            }
        )

    for item in orphan_ends:
        all_items.append(
            {
                "type": "orphan_end",
                "state": "orphan_end",
                "name": item["name"],
                "ts": _round_float(item["ts"]),
                "attempt": item["attempt"],
                "span_id": item.get("span_id"),
                "parent_span_id": item.get("parent_span_id"),
                "parent_span_name": parent_span_name(item.get("parent_span_id")),
                "lane": item["lane"],
                "depth": item["depth"],
                "attrs": _json_safe(item.get("attrs") or {}),
            }
        )

    all_items.sort(
        key=lambda item: (
            item["lane"],
            item.get("start_ts", item.get("ts", 0.0)),
            item.get("display_end_ts", item.get("ts", 0.0)),
            item["name"],
        )
    )

    row_start = min(item.get("start_ts", item.get("ts")) for item in all_items)
    row_end = max(item.get("display_end_ts", item.get("ts")) for item in all_items)
    response_lengths = []
    for item in all_items:
        attrs = item.get("attrs") or {}
        for payload in (attrs, attrs.get("start_attrs"), attrs.get("end_attrs")):
            if not isinstance(payload, dict):
                continue
            response_length = payload.get("response_length")
            if isinstance(response_length, (int, float)):
                response_lengths.append(int(response_length))

    metadata = sample.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    reward = sample.get("reward")
    if isinstance(reward, dict):
        reward = _json_safe(reward)

    return {
        "row_id": sample_idx,
        "sample_index": sample.get("index", sample_idx),
        "group_index": sample.get("group_index"),
        "source": _compact_text(_infer_source(sample, metadata), max_len=64),
        "status": _compact_text(sample.get("status"), max_len=64),
        "label": _compact_text(sample.get("label"), max_len=256),
        "reward": reward,
        "trace_id": _json_safe(trace.get("trace_id")),
        "attempt": int(trace.get("attempt", 0) or 0),
        "start": row_start,
        "end": row_end,
        "duration": _round_float(_safe_duration(row_start, row_end)),
        "lane_count": 1 + max((item["lane"] for item in all_items), default=0),
        "item_count": len(all_items),
        "closed_span_count": sum(1 for item in all_items if item["state"] == "closed_span"),
        "open_span_count": sum(1 for item in all_items if item["state"] == "open_span"),
        "point_event_count": sum(1 for item in all_items if item["state"] == "point_event"),
        "orphan_count": sum(1 for item in all_items if item["state"] == "orphan_end"),
        "total_response_length": sum(response_lengths),
        "max_response_length": max(response_lengths, default=0),
        "items": all_items,
    }


def _build_cache_data(pt_path: Path) -> dict[str, Any]:
    before_missing = len(_MISSING_PICKLE_GLOBALS)
    data = torch.load(
        pt_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_DUMMY_FALLBACK_PICKLE_MODULE,
    )
    if len(_MISSING_PICKLE_GLOBALS) > before_missing:
        missing_names = ", ".join(f"{module}.{name}" for module, name in sorted(_MISSING_PICKLE_GLOBALS))
        print(
            f"[trace_timeline_viewer] substituted missing pickle globals with dummy classes: {missing_names}",
            file=sys.stderr,
        )
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data

    rows: list[dict[str, Any]] = []
    global_start = None
    global_end = None

    for sample_idx, raw_sample in enumerate(samples):
        sample = _to_sample_dict(raw_sample)
        row = _build_items_from_trace(sample, sample_idx)
        if row is None:
            continue
        rows.append(row)
        global_start = row["start"] if global_start is None else min(global_start, row["start"])
        global_end = row["end"] if global_end is None else max(global_end, row["end"])

    return {
        "cache_version": CACHE_VERSION,
        "pt_path": str(pt_path),
        "generated_at": time.time(),
        "sample_count": len(rows),
        "global_start": _round_float(global_start),
        "global_end": _round_float(global_end),
        "rows": rows,
    }


def _timeline_paths(pt_path: Path) -> TimelinePaths:
    stem = pt_path.stem
    directory = pt_path.parent
    return TimelinePaths(
        pt_path=pt_path,
        cache_path=directory / f"{stem}.trace_timeline_cache.json",
        html_path=directory / f"{stem}.trace_timeline_viewer.html",
    )


def ensure_cache(paths: TimelinePaths, rebuild: bool = False) -> dict[str, Any]:
    if not rebuild and paths.cache_path.exists() and paths.cache_path.stat().st_mtime >= paths.pt_path.stat().st_mtime:
        with paths.cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("cache_version") == CACHE_VERSION:
            return cached

    cache_data = _build_cache_data(paths.pt_path)
    with paths.cache_path.open("w", encoding="utf-8") as handle:
        json.dump(cache_data, handle, ensure_ascii=True, separators=(",", ":"))
    return cache_data


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #fffaf2;
      --ink: #22201c;
      --muted: #6f675d;
      --line: #d8cebf;
      --accent: #b5502a;
      --open: #a64f37;
      --point: #2f6d9b;
      --orphan: #a12626;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; overflow: hidden; }
    body {
      margin: 0;
      font-family: "Iosevka Aile", "IBM Plex Sans", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top left, #fffdf8 0, #f6efe2 34%, #f1e8da 100%);
    }
    .page {
      height: 100vh;
      overflow: hidden;
      padding: 18px;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 12px;
    }
    .panel {
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 40px rgba(73, 52, 23, 0.08);
    }
    .controls {
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .controls-header {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
    }
    .controls-extra {
      display: grid;
      gap: 10px;
    }
    .row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    label {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }
    input, select, button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 7px 10px;
      color: var(--ink);
      font: inherit;
    }
    button {
      cursor: pointer;
      background: linear-gradient(180deg, #fffdf9, #f4ebde);
    }
    .btn-subtle {
      padding: 6px 9px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.78);
      color: #6e6457;
    }
    .btn-subtle:hover {
      background: rgba(255, 255, 255, 0.92);
      color: #4f463d;
    }
    .btn-icon {
      min-width: 32px;
      padding: 6px 0;
      text-align: center;
      font-size: 14px;
    }
    .stats { padding: 14px; }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }
    .stat {
      background: rgba(255,255,255,0.66);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
    }
    .stat .name {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .stat .value {
      font-size: 20px;
      font-weight: 700;
    }
    .swatch {
      width: 12px;
      height: 8px;
      border-radius: 2px;
      display: inline-block;
      flex: 0 0 auto;
    }
    .viewport {
      overflow: auto;
      position: relative;
      padding: 0;
      min-height: 0;
      flex: 1 1 auto;
      overscroll-behavior: contain;
    }
    .timeline-panel {
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
    .canvas-wrap {
      position: relative;
      min-height: 120px;
    }
    canvas {
      display: block;
      position: sticky;
      top: 0;
      left: 0;
      z-index: 1;
    }
    .legend {
      padding: 0 14px 14px;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }
    .bar {
      width: 14px;
      height: 8px;
      border-radius: 2px;
      display: inline-block;
    }
    .line {
      width: 14px;
      height: 0;
      border-top: 2px dashed var(--open);
      display: inline-block;
    }
    .cross {
      width: 12px;
      height: 12px;
      display: inline-block;
      position: relative;
    }
    .cross::before, .cross::after {
      content: "";
      position: absolute;
      left: 5px;
      top: 0;
      width: 2px;
      height: 12px;
      background: var(--orphan);
      transform-origin: center;
    }
    .cross::before { transform: rotate(45deg); }
    .cross::after { transform: rotate(-45deg); }
    .tooltip {
      position: fixed;
      z-index: 10;
      max-width: 460px;
      padding: 10px 12px;
      background: rgba(31, 27, 22, 0.94);
      color: #f8f4ec;
      border-radius: 10px;
      pointer-events: none;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity 120ms ease, transform 120ms ease;
      white-space: pre-wrap;
      font-size: 12px;
      line-height: 1.45;
    }
    .tooltip.visible {
      opacity: 1;
      transform: translateY(0);
    }
    .footer {
      padding: 0 14px 14px;
      color: var(--muted);
      font-size: 12px;
    }
    body.compact-ui .controls-extra { display: none; }
  </style>
</head>
<body>
  <div class="page">
    <div class="panel controls">
      <div class="controls-header">
        <div class="row">
          <strong id="title">Trace Timeline</strong>
          <span id="summary" style="color: var(--muted); font-size: 13px;"></span>
        </div>
        <button id="toggleControls" class="btn-subtle btn-icon" type="button">▴</button>
      </div>
      <div class="controls-extra" id="controlsExtra">
        <div class="row">
          <label>Filter <input id="filterText" type="text" placeholder="sample id / source / status / trace"></label>
          <label>Attempt
            <select id="attemptValue">
              <option value="all">all attempt segments</option>
            </select>
          </label>
          <label>Attempt View
            <select id="attemptMode">
              <option value="filter">filter segments</option>
              <option value="highlight">highlight segments</option>
            </select>
          </label>
          <label>View
            <select id="viewMode">
              <option value="collapsed">collapsed</option>
              <option value="expanded">expanded</option>
            </select>
          </label>
          <label>Sort
            <select id="sortMode">
              <option value="start">start</option>
              <option value="duration">duration</option>
              <option value="reward">reward</option>
              <option value="lanes">lane count</option>
              <option value="open">ongoing event count</option>
            </select>
          </label>
          <label><input id="sortDesc" type="checkbox"> desc</label>
          <label><input id="showEvents" type="checkbox" checked> instant events</label>
          <label><input id="showOpenSpans" type="checkbox" checked> ongoing events</label>
          <label><input id="showOrphans" type="checkbox" checked> unmatched ends</label>
          <label>Lane Height <input id="laneHeight" type="range" min="14" max="28" value="18"></label>
          <button id="fitAll" class="btn-subtle" type="button">fit all</button>
          <button id="fitRunning" class="btn-subtle" type="button">fit filtered</button>
        </div>
        <div class="row">
          <span style="color: var(--muted); font-size: 12px;">
            drag = pan, wheel = zoom, click = set cursor and select item
          </span>
          <span id="cursorText" style="color: var(--accent); font-size: 12px;"></span>
        </div>
      </div>
    </div>

    <div class="panel stats">
      <div id="stats" class="stats-grid"></div>
    </div>

    <div class="panel timeline-panel">
      <div id="legend" class="legend"></div>
      <div id="viewport" class="viewport">
        <div id="canvasWrap" class="canvas-wrap">
          <canvas id="timelineCanvas"></canvas>
        </div>
      </div>
      <div class="footer" id="footer"></div>
    </div>
  </div>
  <div id="tooltip" class="tooltip"></div>

  <script>
    const CACHE_FILE = "__CACHE_FILE__";
    const LABEL_WIDTH = 320;
    const AXIS_HEIGHT = 34;
    const ROW_PADDING = 4;
    const state = {
      rawRows: [],
      rows: [],
      rowLayouts: [],
      globalStart: 0,
      globalEnd: 1,
      viewStart: 0,
      viewEnd: 1,
      laneHeight: 18,
      viewMode: 'collapsed',
      cursorTime: null,
      hoveredItem: null,
      selectedItem: null,
      dragging: false,
      dragStartX: 0,
      dragViewStart: 0,
      dragViewEnd: 1,
      rafScheduled: false,
      compactUI: false,
      expandedRows: new Set(),
    };

    const viewport = document.getElementById('viewport');
    const canvasWrap = document.getElementById('canvasWrap');
    const canvas = document.getElementById('timelineCanvas');
    const tooltip = document.getElementById('tooltip');
    const ctx = canvas.getContext('2d');

    function niceDuration(seconds) {
      if (!isFinite(seconds)) return 'n/a';
      if (seconds < 1) return `${(seconds * 1000).toFixed(1)} ms`;
      if (seconds < 60) return `${seconds.toFixed(2)} s`;
      const minutes = Math.floor(seconds / 60);
      const remain = seconds - minutes * 60;
      return `${minutes}m ${remain.toFixed(1)}s`;
    }

    function niceNumber(value) {
      if (value == null || Number.isNaN(value)) return 'n/a';
      if (typeof value === 'number') return value.toLocaleString();
      return `${value}`;
    }

    function escapeHtml(value) {
      return `${value ?? ''}`
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function hashHue(name) {
      let hash = 0;
      for (let idx = 0; idx < name.length; idx += 1) {
        hash = ((hash << 5) - hash + name.charCodeAt(idx)) | 0;
      }
      return ((hash % 360) + 360) % 360;
    }

    function hashColor(name, alpha = 1) {
      const hue = hashHue(name);
      return `hsla(${hue} 55% 46% / ${alpha})`;
    }

    function pointColor(name, alpha = 1) {
      const hue = hashHue(name);
      return `hsla(${hue} 70% 24% / ${alpha})`;
    }

    function pointShape(name) {
      const value = `${name ?? ''}`.toLowerCase();
      if (value.includes('schedule') || value.includes('enqueue') || value.includes('dequeue') || value.includes('buffer')) {
        return 'triangle';
      }
      if (value.includes('assign') || value.includes('complete') || value.includes('selected') || value.includes('finish')) {
        return 'diamond';
      }
      if (value.includes('create') || value.includes('attempt') || value.includes('start')) {
        return 'square';
      }
      return 'circle';
    }

    function itemStart(item) {
      return item.start_ts ?? item.ts ?? 0;
    }

    function itemDisplayEnd(item) {
      return item.display_end_ts ?? item.end_ts ?? item.ts ?? item.start_ts ?? 0;
    }

    function collapsedTextSafeEnd(row, item) {
      let safeEnd = itemDisplayEnd(item);
      if (!item.span_id) return safeEnd;
      for (const candidate of row.items) {
        if (candidate.type !== 'span') continue;
        if (candidate.parent_span_id !== item.span_id) continue;
        const childStart = itemStart(candidate);
        if (childStart > item.start_ts && childStart < safeEnd) {
          safeEnd = childStart;
        }
      }
      return safeEnd;
    }

    function rowLabelParts(row) {
      const sampleId = row.sample_index ?? row.row_id;
      const expandMark = rowExpanded(row) ? '▾' : '▸';
      return `${expandMark} #${sampleId} | ${row.source ?? 'unknown'}`;
    }

    function rowSummaryItem(row) {
      const attempts = [...new Set((row.items || []).map(item => item.attempt).filter(v => v != null))].sort((a, b) => a - b);
      return {
        type: 'row_summary',
        name: 'sample summary',
        attempt: attempts.length > 1 ? attempts.join(',') : (attempts[0] ?? 'n/a'),
        attrs: {
          duration: niceDuration(row.duration ?? 0),
          reward: typeof row.reward === 'number' ? row.reward.toFixed(3) : niceNumber(row.reward),
          attempts: attempts.join(','),
          status: row.status ?? 'unknown',
          label: row.label ?? 'n/a',
          trace_id: row.trace_id ?? 'n/a',
        },
      };
    }

    function sampleMatches(row, filterText) {
      if (!filterText) return true;
      const haystack = [
        row.sample_index,
        row.row_id,
        row.source,
        row.status,
        row.label,
        row.trace_id,
      ].map(v => `${v ?? ''}`.toLowerCase()).join(' ');
      return haystack.includes(filterText.toLowerCase());
    }

    function currentAttemptValue() {
      return document.getElementById('attemptValue').value;
    }

    function currentAttemptMode() {
      return document.getElementById('attemptMode').value;
    }

    function attemptMatches(item) {
      const attemptValue = currentAttemptValue();
      if (attemptValue === 'all') return true;
      return Number(item.attempt) === Number(attemptValue);
    }

    function rowExpanded(row) {
      return state.viewMode === 'expanded' || state.expandedRows.has(row.row_id);
    }

    function rowIsActive(row) {
      return state.selectedItem?.row === row || state.hoveredItem?.row === row;
    }

    function rowAnalysisItems(row) {
      const items = visibleItems(row);
      const attemptValue = currentAttemptValue();
      if (attemptValue === 'all') return items;
      return items.filter(item => attemptMatches(item));
    }

    function visibleItems(row) {
      const showEvents = document.getElementById('showEvents').checked;
      const showOpenSpans = document.getElementById('showOpenSpans').checked;
      const showOrphans = document.getElementById('showOrphans').checked;
      return row.items.filter(item => {
        if (item.state === 'point_event') return showEvents;
        if (item.state === 'open_span') return showOpenSpans;
        if (item.state === 'orphan_end') return showOrphans;
        return true;
      });
    }

    function rowHasMatchingAttempt(row) {
      const attemptValue = currentAttemptValue();
      if (attemptValue === 'all') return true;
      return row.items.some(item => attemptMatches(item));
    }

    function rowVisibleLaneCount(row) {
      const items = rowAnalysisItems(row);
      return 1 + Math.max(0, ...items.map(item => item.lane || 0));
    }

    function itemVisualAlpha(item) {
      const attemptValue = currentAttemptValue();
      const attemptMode = currentAttemptMode();
      if (attemptValue === 'all' || attemptMode !== 'highlight') return 1;
      return attemptMatches(item) ? 1 : 0.18;
    }

    function itemIsFocusedAttempt(item) {
      const attemptValue = currentAttemptValue();
      return attemptValue !== 'all' && attemptMatches(item);
    }

    function drawnItemsForRow(row) {
      return currentAttemptMode() === 'filter' ? rowAnalysisItems(row) : visibleItems(row);
    }

    function buildPointClusters(row, layout, scrollTop, items) {
      const laneGroups = new Map();
      const pointItems = items.filter(item => item.type !== 'span');
      for (const item of pointItems) {
        const lane = rowExpanded(row) ? (item.lane || 0) : 0;
        if (!laneGroups.has(lane)) laneGroups.set(lane, []);
        laneGroups.get(lane).push(item);
      }

      const clusters = [];
      const threshold = 10;
      for (const [lane, laneItems] of laneGroups.entries()) {
        laneItems.sort((a, b) => itemStart(a) - itemStart(b));
        let current = null;
        for (const item of laneItems) {
          const { y, h } = itemGeometry(layout, item, scrollTop);
          const x = timeToX(item.ts ?? item.start_ts ?? 0);
          if (!current || (x - current.maxX) > threshold) {
            current = { lane, items: [item], xSum: x, minX: x, maxX: x, y, h };
            clusters.push(current);
            continue;
          }
          current.items.push(item);
          current.xSum += x;
          current.maxX = x;
        }
      }

      return clusters.map(cluster => {
        const x = cluster.xSum / Math.max(1, cluster.items.length);
        if (cluster.items.length === 1) {
          return { ...cluster, x, item: cluster.items[0], kind: 'single_point' };
        }
        const counts = new Map();
        cluster.items.forEach(item => counts.set(item.name, (counts.get(item.name) || 0) + 1));
        const countEntries = Array.from(counts.entries()).sort((a, b) => {
          if (b[1] !== a[1]) return b[1] - a[1];
          return a[0].localeCompare(b[0]);
        });
        return {
          ...cluster,
          x,
          kind: 'point_cluster',
          item: {
            type: 'point_cluster',
            state: 'point_cluster',
            name: 'point cluster',
            attempt: 'mixed',
            attrs: { total: cluster.items.length, names: Object.fromEntries(countEntries) },
            ts: cluster.items[0].ts ?? cluster.items[0].start_ts ?? 0,
          },
        };
      });
    }

    function legendSpanNames() {
      const counts = new Map();
      for (const row of state.rows) {
        for (const item of rowAnalysisItems(row)) {
          if (item.type !== 'span') continue;
          const name = `${item.name ?? 'event'}`;
          const current = counts.get(name) ?? { name, count: 0 };
          current.count += 1;
          counts.set(name, current);
        }
      }
      return [...counts.values()]
        .sort((a, b) => (b.count - a.count) || a.name.localeCompare(b.name))
        .slice(0, 16);
    }

    function updateLegend() {
      const legend = document.getElementById('legend');
      if (!legend) return;
      const names = legendSpanNames();
      if (!names.length) {
        legend.innerHTML = '<span class="chip">no span events</span>';
        return;
      }
      // Check whether any row has PD data
      let hasPD = false;
      outer: for (const row of state.rows) {
        for (const item of rowAnalysisItems(row)) {
          if (item.type === 'span' && pdPhases(spanFlatAttrs(item))) { hasPD = true; break outer; }
        }
      }
      let html = names.map((entry) => (
        `<span class="chip" title="count=${entry.count}">` +
        `<span class="bar" style="background: ${hashColor(entry.name, 0.9)}"></span>` +
        `${escapeHtml(entry.name)}` +
        `</span>`
      )).join('');
      if (hasPD) {
        const pdLegend = [
          ['prefill_forward', 'prefill fwd'],
          ['decode_forward', 'decode fwd'],
          ['prefill_transfer_queue', 'transfer'],
          ['prefill_bootstrap_queue', 'bootstrap queue'],
          ['decode_prealloc', 'prealloc'],
          ['decode_transfer', 'kv transfer'],
          ['bootstrap', 'bootstrap'],
          ['alloc_waiting', 'alloc wait'],
        ];
        html += '<span style="color:var(--muted);margin-left:8px;font-size:11px">PD:</span>';
        for (const [phase, label] of pdLegend) {
          html += `<span class="chip" style="font-size:11px">` +
            `<span class="bar" style="background:${pdPhaseColor(phase, 0.85)}"></span>` +
            `${label}</span>`;
        }
      }
      legend.innerHTML = html;
    }

    function applyFilterAndSort() {
      const filterText = document.getElementById('filterText').value.trim();
      const sortMode = document.getElementById('sortMode').value;
      const sortDesc = document.getElementById('sortDesc').checked;
      state.rows = state.rawRows.filter(row => sampleMatches(row, filterText));
      if (currentAttemptMode() === 'filter') {
        state.rows = state.rows.filter(row => rowHasMatchingAttempt(row));
      }
      state.rows.sort((a, b) => {
        let delta = (a.start ?? 0) - (b.start ?? 0);
        if (sortMode === 'duration') delta = (a.duration ?? 0) - (b.duration ?? 0);
        if (sortMode === 'reward') delta = (Number(a.reward) || 0) - (Number(b.reward) || 0);
        if (sortMode === 'lanes') delta = (a.lane_count ?? 0) - (b.lane_count ?? 0);
        if (sortMode === 'open') delta = (a.open_span_count ?? 0) - (b.open_span_count ?? 0);
        return sortDesc ? -delta : delta;
      });
      layout();
      updateLegend();
      updateStats();
      scheduleDraw();
    }

    function computeRowLayouts() {
      const layouts = [];
      let cursor = AXIS_HEIGHT;
      for (const row of state.rows) {
        const collapsedHeight = state.laneHeight + ROW_PADDING * 2;
        const expandedHeight = rowVisibleLaneCount(row) * state.laneHeight + ROW_PADDING * 2;
        const rowHeight = Math.max(
          state.laneHeight + ROW_PADDING * 2,
          rowExpanded(row) ? expandedHeight : collapsedHeight,
        );
        layouts.push({ top: cursor, height: rowHeight });
        cursor += rowHeight;
      }
      state.rowLayouts = layouts;
      return cursor;
    }

    function layout() {
      const totalHeight = computeRowLayouts();
      canvasWrap.style.height = `${Math.max(totalHeight + 8, 120)}px`;
      const width = Math.max(720, viewport.clientWidth - 2);
      const height = Math.max(240, viewport.clientHeight - 2);
      canvas.width = width * window.devicePixelRatio;
      canvas.height = height * window.devicePixelRatio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
      updateSummary();
    }

    function updateSummary() {
      document.getElementById('summary').textContent =
        `${state.rows.length} filtered / ${state.rawRows.length} total`;
      document.getElementById('footer').textContent =
        `view range: ${niceDuration(state.viewEnd - state.viewStart)} | global range: ${niceDuration(state.globalEnd - state.globalStart)}`;
    }

    function timeToX(timestamp) {
      const width = canvas.clientWidth - LABEL_WIDTH;
      return LABEL_WIDTH + ((timestamp - state.viewStart) / (state.viewEnd - state.viewStart)) * width;
    }

    function xToTime(x) {
      const width = canvas.clientWidth - LABEL_WIDTH;
      const ratio = (x - LABEL_WIDTH) / width;
      return state.viewStart + ratio * (state.viewEnd - state.viewStart);
    }

    function laneTop(layout, lane, scrollTop) {
      return layout.top - scrollTop + ROW_PADDING + lane * state.laneHeight;
    }

    function itemGeometry(layout, item, scrollTop) {
      const rowIdx = state.rowLayouts.indexOf(layout);
      const row = rowIdx >= 0 ? state.rows[rowIdx] : null;
      if (row && rowExpanded(row)) {
        return {
          y: laneTop(layout, item.lane || 0, scrollTop) + 2,
          h: Math.max(10, state.laneHeight - 4),
        };
      }

      const baseY = layout.top - scrollTop + ROW_PADDING + 2;
      return {
        y: baseY,
        h: Math.max(10, layout.height - ROW_PADDING * 2 - 4),
      };
    }

    function drawAxis() {
      const width = canvas.clientWidth;
      const height = canvas.height / window.devicePixelRatio;
      ctx.fillStyle = '#fffaf2';
      ctx.fillRect(0, 0, width, AXIS_HEIGHT);
      ctx.fillStyle = '#6f675d';
      ctx.font = '12px IBM Plex Sans, sans-serif';
      ctx.textBaseline = 'top';
      ctx.fillText('sample', 12, 10);
      const range = state.viewEnd - state.viewStart;
      const targetTicks = 8;
      const rawStep = range / targetTicks;
      const power = Math.pow(10, Math.floor(Math.log10(Math.max(rawStep, 1e-6))));
      const candidates = [1, 2, 5, 10];
      let step = power;
      for (const candidate of candidates) {
        const maybe = candidate * power;
        if (maybe >= rawStep) {
          step = maybe;
          break;
        }
      }
      const firstTick = Math.ceil(state.viewStart / step) * step;
      ctx.strokeStyle = '#d8cebf';
      ctx.fillStyle = '#6f675d';
      for (let tick = firstTick; tick <= state.viewEnd; tick += step) {
        const x = timeToX(tick);
        ctx.beginPath();
        ctx.moveTo(x, AXIS_HEIGHT - 12);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.fillText(niceDuration(tick - state.globalStart), x + 4, 10);
      }
      ctx.beginPath();
      ctx.moveTo(0, AXIS_HEIGHT - 0.5);
      ctx.lineTo(width, AXIS_HEIGHT - 0.5);
      ctx.strokeStyle = '#bdb3a3';
      ctx.stroke();
    }

    function lowerBoundRows(targetY) {
      let left = 0;
      let right = state.rowLayouts.length;
      while (left < right) {
        const mid = (left + right) >> 1;
        const layout = state.rowLayouts[mid];
        if (layout.top + layout.height < targetY) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
      return left;
    }

    function visibleRowRange() {
      const width = canvas.clientWidth;
      const visibleStartY = viewport.scrollTop;
      const visibleEndY = visibleStartY + viewport.clientHeight;
      const startRow = Math.max(0, lowerBoundRows(visibleStartY) - 1);
      let endRow = startRow;
      while (endRow < state.rowLayouts.length && state.rowLayouts[endRow].top <= visibleEndY + state.laneHeight * 2) {
        endRow += 1;
      }
      return { width, startRow, endRow: Math.max(startRow - 1, endRow - 1) };
    }

    // PD phase color palette (warm earth tones matching the viewer aesthetic)
    const PD_COLORS = {
      prefill_bootstrap_queue: 'rgba(180, 160, 120, A)',  // muted sand
      bootstrap:               'rgba(140, 120, 90, A)',   // dark sand
      alloc_waiting:           'rgba(160, 140, 110, A)',  // warm grey
      prefill_forward:         'rgba(70, 140, 180, A)',   // steel blue (prefill)
      prefill_transfer_queue:  'rgba(200, 160, 60, A)',   // amber (transfer)
      decode_prealloc:         'rgba(160, 140, 110, A)',  // warm grey
      decode_transfer:         'rgba(200, 160, 60, A)',   // amber (transfer)
      decode_forward:          'rgba(80, 170, 100, A)',   // green (decode)
    };
    function pdPhaseColor(phase, alpha) {
      return (PD_COLORS[phase] || 'rgba(128,128,128,A)').replace(/A/g, alpha.toFixed(2));
    }

    // For span items, attrs is {start_attrs, end_attrs}. Flatten into a single dict.
    function spanFlatAttrs(item) {
      const raw = item.attrs || {};
      if (raw.start_attrs || raw.end_attrs) {
        return Object.assign({}, raw.start_attrs || {}, raw.end_attrs || {});
      }
      return raw;
    }

    // Build ordered PD phase segments from span attrs.
    // Returns [{phase, duration}] or null if no PD data.
    function pdPhases(attrs) {
      if (!attrs) return null;
      const segs = [];
      const push = (phase, key) => {
        const v = attrs[`pd_${key}`];
        if (v != null && v > 0) segs.push({ phase, duration: v });
      };
      // P-side phases (sequential order)
      push('prefill_bootstrap_queue', 'prefill_bootstrap_queue_duration');
      push('bootstrap',               'bootstrap_duration');
      push('alloc_waiting',           'alloc_waiting_duration');
      push('prefill_forward',         'prefill_forward_duration');
      push('prefill_transfer_queue',  'prefill_transfer_queue_duration');
      // D-side phases (sequential order)
      push('decode_prealloc',  'decode_prealloc_duration');
      push('decode_transfer',  'decode_transfer_duration');
      push('decode_forward',   'decode_forward_duration');
      return segs.length > 0 ? segs : null;
    }

    function drawSpan(item, x1, x2, y, h) {
      const alpha = itemVisualAlpha(item);
      const color = hashColor(item.name, item.state === 'open_span' ? 0.25 * alpha : 0.9 * alpha);
      const border = hashColor(item.name, Math.max(0.22, alpha));
      const spanWidth = Math.max(1.5, x2 - x1);

      if (item.state === 'open_span') {
        ctx.save();
        ctx.fillStyle = color;
        ctx.fillRect(x1, y, Math.max(2, x2 - x1), h);
        ctx.strokeStyle = border;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x1 + 0.5, y + 0.5, Math.max(1, x2 - x1 - 1), Math.max(1, h - 1));
        ctx.setLineDash([]);
        ctx.restore();
        const fadeWidth = Math.min(18, Math.max(6, x2 - x1));
        const gradient = ctx.createLinearGradient(x2 - fadeWidth, 0, x2, 0);
        gradient.addColorStop(0, hashColor(item.name, 0));
        gradient.addColorStop(1, hashColor(item.name, Math.max(0.22, alpha)));
        ctx.fillStyle = gradient;
        ctx.fillRect(Math.max(x1, x2 - fadeWidth), y, fadeWidth, h);
        if (itemIsFocusedAttempt(item)) {
          ctx.save();
          ctx.strokeStyle = '#1f4b3a';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x1 + 0.5, y + 0.5, Math.max(1, x2 - x1 - 1), Math.max(1, h - 1));
          ctx.restore();
        }
        return;
      }

      // Draw PD phase sub-bars when disaggregation data is available
      const phases = pdPhases(spanFlatAttrs(item));
      if (phases && spanWidth > 6) {
        const totalPhaseDuration = phases.reduce((s, p) => s + p.duration, 0);
        // Draw base bar at reduced opacity
        ctx.fillStyle = hashColor(item.name, 0.15 * alpha);
        ctx.fillRect(x1, y, spanWidth, h);

        // Draw each phase as a sub-segment
        const subH = Math.max(4, h - 4);
        const subY = y + 2;
        let cx = x1;
        for (const seg of phases) {
          const w = Math.max(1, (seg.duration / totalPhaseDuration) * spanWidth);
          ctx.fillStyle = pdPhaseColor(seg.phase, 0.85 * alpha);
          ctx.fillRect(cx, subY, w, subH);
          // thin separator
          if (cx > x1) {
            ctx.fillStyle = 'rgba(0,0,0,0.25)';
            ctx.fillRect(cx, subY, 0.5, subH);
          }
          cx += w;
        }

        // Border around the full span
        ctx.strokeStyle = border;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x1 + 0.5, y + 0.5, Math.max(1, spanWidth - 1), Math.max(1, h - 1));
      } else {
        ctx.fillStyle = color;
        ctx.fillRect(x1, y, spanWidth, h);
      }

      if (itemIsFocusedAttempt(item)) {
        ctx.save();
        ctx.strokeStyle = '#1f4b3a';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x1 + 0.5, y + 0.5, Math.max(1, x2 - x1 - 1), Math.max(1, h - 1));
        ctx.restore();
      }
    }

    function drawPoint(item, x, y, h) {
      const midY = y + h / 2;
      if (item.state === 'orphan_end') {
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--orphan').trim();
        ctx.globalAlpha = itemVisualAlpha(item);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - 4, midY - 4);
        ctx.lineTo(x + 4, midY + 4);
        ctx.moveTo(x + 4, midY - 4);
        ctx.lineTo(x - 4, midY + 4);
        ctx.stroke();
        ctx.globalAlpha = 1;
        return;
      }
      const color = pointColor(item.name, itemVisualAlpha(item));
      const shape = pointShape(item.name);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x, y + h);
      ctx.stroke();
      ctx.fillStyle = color;
      if (shape === 'triangle') {
        ctx.beginPath();
        ctx.moveTo(x, midY - 5);
        ctx.lineTo(x + 5, midY + 4);
        ctx.lineTo(x - 5, midY + 4);
        ctx.closePath();
        ctx.fill();
        return;
      }
      if (shape === 'square') {
        ctx.fillRect(x - 4, midY - 4, 8, 8);
        return;
      }
      if (shape === 'circle') {
        ctx.beginPath();
        ctx.arc(x, midY, 4.5, 0, Math.PI * 2);
        ctx.fill();
        return;
      }
      ctx.beginPath();
      ctx.moveTo(x, midY - 5);
      ctx.lineTo(x + 5, midY);
      ctx.lineTo(x, midY + 5);
      ctx.lineTo(x - 5, midY);
      ctx.closePath();
      ctx.fill();
    }

    function drawPointCluster(cluster) {
      if (cluster.kind === 'single_point') {
        drawPoint(cluster.item, cluster.x, cluster.y, cluster.h);
        return;
      }
      const midY = cluster.y + cluster.h / 2;
      ctx.fillStyle = 'rgba(32, 28, 24, 0.88)';
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(cluster.x, midY, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#f7f2ea';
      ctx.font = '10px IBM Plex Sans, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const label = cluster.items.length > 9 ? '9+' : `${cluster.items.length}`;
      ctx.fillText(label, cluster.x, midY + 0.5);
      ctx.textAlign = 'start';
      ctx.textBaseline = 'alphabetic';
    }

    function drawRows() {
      const { width, startRow, endRow } = visibleRowRange();
      if (endRow < startRow) return;
      const scrollTop = viewport.scrollTop;
      const showText = (state.viewEnd - state.viewStart) <= 120;

      for (let rowIdx = startRow; rowIdx <= endRow; rowIdx += 1) {
        const row = state.rows[rowIdx];
        const layout = state.rowLayouts[rowIdx];
        if (!row || !layout) continue;
        const top = layout.top - scrollTop;
        const activeRow = rowIsActive(row);
        const expandedRow = rowExpanded(row) && state.viewMode !== 'expanded';
        ctx.fillStyle = activeRow
          ? 'rgba(181, 80, 42, 0.12)'
          : expandedRow
            ? 'rgba(47, 109, 155, 0.08)'
            : rowIdx % 2 === 0
              ? 'rgba(216, 206, 191, 0.10)'
              : 'rgba(216, 206, 191, 0.03)';
        ctx.fillRect(0, top, width, layout.height);
        if (expandedRow) {
          ctx.fillStyle = 'rgba(47, 109, 155, 0.10)';
          ctx.fillRect(0, top, LABEL_WIDTH, layout.height);
        }
        const label = rowLabelParts(row);
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, top, LABEL_WIDTH - 8, layout.height);
        ctx.clip();
        ctx.textBaseline = 'middle';
        ctx.fillStyle = activeRow ? '#6c341f' : '#5f574d';
        ctx.font = '11px IBM Plex Sans, sans-serif';
        ctx.fillText(label, 12, top + layout.height / 2);
        ctx.restore();

        if (rowExpanded(row)) {
          for (let lane = 1; lane < rowVisibleLaneCount(row); lane += 1) {
            const y = laneTop(layout, lane, scrollTop) - 1;
            ctx.strokeStyle = 'rgba(189, 179, 163, 0.35)';
            ctx.beginPath();
            ctx.moveTo(LABEL_WIDTH, y);
            ctx.lineTo(width, y);
            ctx.stroke();
          }
        }

        const drawnItems = drawnItemsForRow(row);
        const spanItems = drawnItems.filter(item => item.type === 'span');
        const pointClusters = buildPointClusters(row, layout, scrollTop, drawnItems);
        ctx.save();
        ctx.beginPath();
        ctx.rect(LABEL_WIDTH, top, Math.max(1, width - LABEL_WIDTH), layout.height);
        ctx.clip();
        for (const item of spanItems) {
          const { y, h } = itemGeometry(layout, item, scrollTop);
          const start = itemStart(item);
          const end = itemDisplayEnd(item);
          if (end < state.viewStart || start > state.viewEnd) continue;
          if (item.type === 'span') {
            const x1 = Math.max(LABEL_WIDTH, timeToX(start));
            const x2 = Math.min(width, timeToX(end));
            drawSpan(item, x1, x2, y, h);
            if (state.selectedItem?.item === item) {
              ctx.save();
              ctx.strokeStyle = '#b5502a';
              ctx.lineWidth = 2;
              ctx.strokeRect(x1 + 0.5, y + 0.5, Math.max(1, x2 - x1 - 1), Math.max(1, h - 1));
              ctx.restore();
            }
            const labelEnd = !rowExpanded(row)
              ? Math.min(end, collapsedTextSafeEnd(row, item))
              : end;
            const labelX2 = Math.min(width, timeToX(labelEnd));
            const labelWidthAvailable = labelX2 - x1;
            if (showText && labelWidthAvailable > 40) {
              ctx.font = '10px IBM Plex Sans, sans-serif';
              const label = (() => {
                const a = spanFlatAttrs(item);
                const pdParts = [];
                if (a.prompt_tokens != null) pdParts.push(`P:${a.prompt_tokens}`);
                if (a.completion_tokens != null) pdParts.push(`D:${a.completion_tokens}`);
                if (a.cached_tokens != null && a.cached_tokens > 0) pdParts.push(`cache:${a.cached_tokens}`);
                if (a.pd_prefill_forward_duration != null) pdParts.push(`pf:${(a.pd_prefill_forward_duration*1000).toFixed(0)}ms`);
                if (a.pd_decode_forward_duration != null) pdParts.push(`df:${(a.pd_decode_forward_duration*1000).toFixed(0)}ms`);
                if (a.pd_transfer_speed_gb_s != null) pdParts.push(`${a.pd_transfer_speed_gb_s.toFixed(1)}GB/s`);
                const pdSuffix = pdParts.length > 0 ? ` | ${pdParts.join(' ')}` : '';
                return `${item.name} | attempt=${item.attempt}${pdSuffix}`;
              })();
              const labelWidth = ctx.measureText(label).width;
              if (labelWidth + 12 <= labelWidthAvailable) {
                ctx.fillStyle = itemVisualAlpha(item) < 1 ? 'rgba(255,255,255,0.45)' : 'rgba(255,255,255,0.86)';
                ctx.font = '10px IBM Plex Sans, sans-serif';
                ctx.save();
                ctx.beginPath();
                ctx.rect(x1, y, labelWidthAvailable, h);
                ctx.clip();
                ctx.textBaseline = 'middle';
                ctx.fillText(label, x1 + 4, y + h / 2 + 0.5);
                ctx.restore();
              }
            }
            continue;
          }
        }
        pointClusters.forEach(cluster => drawPointCluster(cluster));
        ctx.restore();
      }
    }

    function drawCursor() {
      if (state.cursorTime == null) return;
      if (state.cursorTime < state.viewStart || state.cursorTime > state.viewEnd) return;
      const x = timeToX(state.cursorTime);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height / window.devicePixelRatio);
      ctx.lineWidth = 1;
      ctx.strokeStyle = '#b5502a';
      ctx.stroke();
    }

    function itemTooltipLines(row, item) {
      const lines = [
        `sample: ${row.sample_index ?? row.row_id}`,
        `source: ${row.source ?? 'unknown'}`,
        `name: ${item.name}`,
        `attempt: ${item.attempt}`,
      ];
      if (item.type === 'row_summary') {
        for (const [key, value] of Object.entries(item.attrs || {})) {
          lines.push(`${key}: ${value}`);
        }
        return lines;
      }
      if (item.type === 'point_cluster') {
        lines.push(`events: ${item.attrs?.total ?? 0}`);
        Object.entries(item.attrs?.names || {}).slice(0, 8).forEach(([name, count]) => {
          lines.push(`${name}: ${count}`);
        });
        return lines;
      }
      if (item.parent_span_name) lines.push(`parent: ${item.parent_span_name}`);
      if (item.type === 'span') {
        lines.push(`start: ${niceDuration(item.start_ts - state.globalStart)}`);
        lines.push(`end: ${item.end_ts == null ? 'open' : niceDuration(item.end_ts - state.globalStart)}`);
        if (item.end_ts == null) {
          lines.push(`display_end: ${niceDuration(item.display_end_ts - state.globalStart)}`);
        }
        lines.push(`duration: ${item.end_ts == null ? 'open' : niceDuration(item.end_ts - item.start_ts)}`);
      } else {
        lines.push(`time: ${niceDuration((item.ts ?? 0) - state.globalStart)}`);
      }
      if (item.state === 'orphan_end') lines.push('warning: unmatched end event');
      const attrs = spanFlatAttrs(item);
      // Collect and group PD disaggregation attrs for structured display
      const pdKeys = Object.keys(attrs).filter(k => k.startsWith('pd_'));
      const otherKeys = Object.keys(attrs).filter(k => !k.startsWith('pd_'));
      for (const key of otherKeys) {
        const value = attrs[key];
        lines.push(`${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`);
      }
      if (pdKeys.length > 0) {
        const phases = pdPhases(attrs);
        if (phases) {
          const totalDur = phases.reduce((s, p) => s + p.duration, 0);
          const pPhases = phases.filter(p => p.phase.startsWith('prefill_') || p.phase === 'bootstrap' || p.phase === 'alloc_waiting');
          const dPhases = phases.filter(p => p.phase.startsWith('decode_'));
          const fmtPhase = (p) => {
            const ms = (p.duration * 1000).toFixed(1);
            const pct = totalDur > 0 ? ((p.duration / totalDur) * 100).toFixed(0) : 0;
            const nice = p.phase.replace(/_/g, ' ');
            return `  ${nice}: ${ms}ms (${pct}%)`;
          };
          lines.push('── PD disaggregation ──');
          if (pPhases.length > 0) {
            lines.push(' [P] prefill instance');
            pPhases.forEach(p => lines.push(fmtPhase(p)));
          }
          if (dPhases.length > 0) {
            lines.push(' [D] decode instance');
            dPhases.forEach(p => lines.push(fmtPhase(p)));
          }
          lines.push(`  total phases: ${(totalDur * 1000).toFixed(1)}ms`);
        }
        // Show non-duration PD fields (transfer speed, total MB, retry count)
        for (const key of pdKeys) {
          const value = attrs[key];
          const label = key.replace(/^pd_/, '');
          if (label.endsWith('_duration')) continue;  // already shown above
          if (label.endsWith('_gb_s') && typeof value === 'number') {
            lines.push(`  ${label}: ${value.toFixed(2)} GB/s`);
          } else if (label.endsWith('_mb') && typeof value === 'number') {
            lines.push(`  ${label}: ${value.toFixed(1)} MB`);
          } else {
            lines.push(`  ${label}: ${typeof value === 'object' ? JSON.stringify(value) : value}`);
          }
        }
      }
      return lines;
    }

    function updateTooltip() {
      const activeItem = state.hoveredItem || state.selectedItem;
      if (!activeItem) {
        tooltip.classList.remove('visible');
        return;
      }
      const { row, item, x, y } = activeItem;
      tooltip.textContent = itemTooltipLines(row, item).join('\n');
      tooltip.style.left = `${Math.min(window.innerWidth - 480, x + 14)}px`;
      tooltip.style.top = `${Math.min(window.innerHeight - 200, y + 14)}px`;
      tooltip.classList.add('visible');
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.height / window.devicePixelRatio);
      drawRows();
      drawAxis();
      drawCursor();
      updateTooltip();
      updateSummary();
    }

    function scheduleDraw() {
      if (state.rafScheduled) return;
      state.rafScheduled = true;
      requestAnimationFrame(() => {
        state.rafScheduled = false;
        draw();
      });
    }

    function panTimelineByPixels(deltaX) {
      const width = Math.max(1, canvas.clientWidth - LABEL_WIDTH);
      const shift = (deltaX / width) * (state.viewEnd - state.viewStart);
      let nextStart = state.viewStart + shift;
      let nextEnd = state.viewEnd + shift;
      if (nextStart < state.globalStart) {
        nextEnd += state.globalStart - nextStart;
        nextStart = state.globalStart;
      }
      if (nextEnd > state.globalEnd) {
        nextStart -= nextEnd - state.globalEnd;
        nextEnd = state.globalEnd;
      }
      state.viewStart = nextStart;
      state.viewEnd = nextEnd;
      scheduleDraw();
    }

    function setCompactUI(nextCompact) {
      state.compactUI = nextCompact;
      document.body.classList.toggle('compact-ui', nextCompact);
      document.getElementById('toggleControls').textContent = nextCompact ? '▾' : '▴';
      layout();
      scheduleDraw();
    }

    function fitToRows(rows) {
      if (!rows.length) return;
      let minStart = rows[0].start;
      let maxEnd = rows[0].end;
      rows.forEach(row => {
        minStart = Math.min(minStart, row.start);
        maxEnd = Math.max(maxEnd, row.end);
      });
      const pad = Math.max((maxEnd - minStart) * 0.03, 0.001);
      state.viewStart = minStart - pad;
      state.viewEnd = maxEnd + pad;
    }

    function updateStats() {
      const lifecycleStats = {
        rows: state.rows.length,
        notStarted: 0,
        running: 0,
        ended: 0,
      };
      const nameCounts = new Map();
      const cursorTime = state.cursorTime ?? state.viewStart;
      const pointTimeWindow = (state.viewEnd - state.viewStart) * 6 / Math.max(1, canvas.clientWidth - LABEL_WIDTH);

      function addCount(name) {
        nameCounts.set(name, (nameCounts.get(name) || 0) + 1);
      }

      state.rows.forEach(row => {
        if (cursorTime < row.start) {
          lifecycleStats.notStarted += 1;
          return;
        }
        if (cursorTime > row.end) {
          lifecycleStats.ended += 1;
          return;
        }
        lifecycleStats.running += 1;

        for (const item of rowAnalysisItems(row)) {
          if (item.type === 'span') {
            if (cursorTime >= item.start_ts && cursorTime <= itemDisplayEnd(item)) {
              addCount(item.name);
            }
            continue;
          }
          const ts = item.ts ?? 0;
          if (Math.abs(ts - cursorTime) <= pointTimeWindow) {
            addCount(item.name);
          }
        }
      });
      document.getElementById('cursorText').textContent =
        `cursor: ${niceDuration(cursorTime - state.globalStart)}`;
      const lifecycleItems = [
        { name: 'total samples', value: lifecycleStats.rows, kind: 'lifecycle' },
        { name: 'not started', value: lifecycleStats.notStarted, kind: 'lifecycle' },
        { name: 'running', value: lifecycleStats.running, kind: 'lifecycle' },
        { name: 'ended', value: lifecycleStats.ended, kind: 'lifecycle' },
      ];
      const eventItems = Array.from(nameCounts.entries())
        .sort((a, b) => {
          if (b[1] !== a[1]) return b[1] - a[1];
          return a[0].localeCompare(b[0]);
        })
        .map(([name, value]) => ({ name, value, kind: 'event' }));
      const items = lifecycleItems.concat(eventItems);
      document.getElementById('stats').innerHTML = items.map((item) => `
        <div class="stat">
          <div class="name">${
            item.kind === 'event'
              ? `<span class="swatch" style="background:${hashColor(item.name, 0.9)}"></span>${escapeHtml(item.name)}`
              : escapeHtml(item.name)
          }</div>
          <div class="value">${item.value}</div>
        </div>
      `).join('');
    }

    function handleZoom(mouseX, deltaY) {
      const focusTime = xToTime(mouseX);
      const range = state.viewEnd - state.viewStart;
      const scale = deltaY < 0 ? 0.9 : 1.1;
      const newRange = Math.min(state.globalEnd - state.globalStart, Math.max(0.01, range * scale));
      const ratio = (focusTime - state.viewStart) / range;
      let nextStart = focusTime - ratio * newRange;
      let nextEnd = nextStart + newRange;
      if (nextStart < state.globalStart) {
        nextEnd += state.globalStart - nextStart;
        nextStart = state.globalStart;
      }
      if (nextEnd > state.globalEnd) {
        nextStart -= nextEnd - state.globalEnd;
        nextEnd = state.globalEnd;
      }
      state.viewStart = nextStart;
      state.viewEnd = nextEnd;
      scheduleDraw();
    }

    function rowIndexForCanvasY(canvasY) {
      const absoluteY = canvasY + viewport.scrollTop;
      const idx = lowerBoundRows(absoluteY);
      if (idx >= state.rowLayouts.length) return -1;
      const layout = state.rowLayouts[idx];
      if (absoluteY >= layout.top && absoluteY <= layout.top + layout.height) return idx;
      return -1;
    }

    function findItemAt(row, rowIdx, canvasX, canvasY) {
      const layout = state.rowLayouts[rowIdx];
      if (!layout) return null;
      const localY = canvasY + viewport.scrollTop - layout.top;
      const cursorTime = xToTime(canvasX);
      const pointPixelTolerance = 8;
      const hitItems = drawnItemsForRow(row);
      const pointClusters = buildPointClusters(row, layout, 0, hitItems);
      let closestPoint = null;
      let closestPointDistance = Infinity;

      for (const cluster of pointClusters) {
        const pointY = (cluster.y - layout.top) + cluster.h / 2;
        const dx = canvasX - cluster.x;
        const dy = localY - pointY;
        const radius = cluster.kind === 'point_cluster' ? 9 : pointPixelTolerance;
        const distance = Math.hypot(dx, dy);
        if (distance <= radius && distance < closestPointDistance) {
          closestPoint = cluster.item;
          closestPointDistance = distance;
        }
      }

      if (closestPoint) return closestPoint;

      for (const item of [...hitItems].reverse()) {
        if (item.type !== 'span') continue;
        const geometry = itemGeometry(layout, item, 0);
        const itemTop = geometry.y - layout.top;
        const itemBottom = itemTop + geometry.h;
        if (localY < itemTop || localY > itemBottom) continue;
        if (cursorTime >= item.start_ts && cursorTime <= itemDisplayEnd(item)) return item;
      }

      return null;
    }

    function attachEvents() {
      document.getElementById('filterText').addEventListener('input', applyFilterAndSort);
      document.getElementById('attemptValue').addEventListener('change', applyFilterAndSort);
      document.getElementById('attemptMode').addEventListener('change', applyFilterAndSort);
      document.getElementById('viewMode').addEventListener('change', (event) => {
        state.viewMode = event.target.value;
        layout();
        updateLegend();
        updateStats();
        scheduleDraw();
      });
      document.getElementById('sortMode').addEventListener('change', applyFilterAndSort);
      document.getElementById('sortDesc').addEventListener('change', applyFilterAndSort);
      document.getElementById('showEvents').addEventListener('change', () => { updateLegend(); updateStats(); scheduleDraw(); });
      document.getElementById('showOpenSpans').addEventListener('change', () => { layout(); updateLegend(); updateStats(); scheduleDraw(); });
      document.getElementById('showOrphans').addEventListener('change', () => { updateLegend(); updateStats(); scheduleDraw(); });
      document.getElementById('toggleControls').addEventListener('click', () => setCompactUI(!state.compactUI));
      document.getElementById('laneHeight').addEventListener('input', (event) => {
        state.laneHeight = Number(event.target.value);
        layout();
        scheduleDraw();
      });
      document.getElementById('fitAll').addEventListener('click', () => {
        fitToRows(state.rawRows);
        scheduleDraw();
      });
      document.getElementById('fitRunning').addEventListener('click', () => {
        fitToRows(state.rows);
        scheduleDraw();
      });
      viewport.addEventListener('scroll', () => scheduleDraw());
      canvas.addEventListener('mousemove', (event) => {
        if (state.dragging) {
          const dx = event.clientX - state.dragStartX;
          const dy = event.clientY - state.dragStartY;
          if (!state.dragMoved && Math.hypot(dx, dy) > 3) {
            state.dragMoved = true;
          }
          const pxToTime = (state.dragViewEnd - state.dragViewStart) / (canvas.clientWidth - LABEL_WIDTH);
          const shift = dx * pxToTime;
          let nextStart = state.dragViewStart - shift;
          let nextEnd = state.dragViewEnd - shift;
          if (nextStart < state.globalStart) {
            nextEnd += state.globalStart - nextStart;
            nextStart = state.globalStart;
          }
          if (nextEnd > state.globalEnd) {
            nextStart -= nextEnd - state.globalEnd;
            nextEnd = state.globalEnd;
          }
          state.viewStart = nextStart;
          state.viewEnd = nextEnd;
          scheduleDraw();
          return;
        }
        const rect = canvas.getBoundingClientRect();
        const canvasX = event.clientX - rect.left;
        const canvasY = event.clientY - rect.top;
        const rowIdx = rowIndexForCanvasY(canvasY);
        if (canvasX < LABEL_WIDTH) {
          let hoveredItem = null;
          if (rowIdx >= 0 && rowIdx < state.rows.length) {
            const row = state.rows[rowIdx];
            hoveredItem = { row, item: rowSummaryItem(row), x: event.clientX, y: event.clientY };
          }
          const sameHovered = state.hoveredItem
            && hoveredItem
            && state.hoveredItem.row === hoveredItem.row
            && state.hoveredItem.item.type === hoveredItem.item.type;
          if (!sameHovered || (!hoveredItem && state.hoveredItem)) {
            state.hoveredItem = hoveredItem;
            updateTooltip();
          } else if (hoveredItem) {
            state.hoveredItem = hoveredItem;
            updateTooltip();
          }
          return;
        }
        let hoveredItem = null;
        if (rowIdx >= 0 && rowIdx < state.rows.length) {
          const row = state.rows[rowIdx];
          const item = findItemAt(row, rowIdx, canvasX, canvasY);
          if (item && item.type !== 'span') hoveredItem = { row, item, x: event.clientX, y: event.clientY };
        }
        const sameHovered = state.hoveredItem
          && hoveredItem
          && state.hoveredItem.item === hoveredItem.item
          && state.hoveredItem.row === hoveredItem.row;
        if (!sameHovered || (!hoveredItem && state.hoveredItem)) {
          state.hoveredItem = hoveredItem;
          updateTooltip();
        } else if (hoveredItem) {
          state.hoveredItem = hoveredItem;
          updateTooltip();
        }
      });
      canvas.addEventListener('mouseleave', () => {
        if (state.hoveredItem) {
          state.hoveredItem = null;
          updateTooltip();
        }
      });
      canvas.addEventListener('mousedown', (event) => {
        event.preventDefault();
        state.dragging = true;
        state.dragStartX = event.clientX;
        state.dragViewStart = state.viewStart;
        state.dragViewEnd = state.viewEnd;
      });
      window.addEventListener('mouseup', () => { state.dragging = false; });
      canvas.addEventListener('click', (event) => {
        const rect = canvas.getBoundingClientRect();
        const canvasX = event.clientX - rect.left;
        const canvasY = event.clientY - rect.top;
        const rowIdx = rowIndexForCanvasY(canvasY);
        if (canvasX < LABEL_WIDTH) {
          if (rowIdx >= 0 && rowIdx < state.rows.length && state.viewMode !== 'expanded') {
            const row = state.rows[rowIdx];
            state.selectedItem = { row, item: rowSummaryItem(row), x: event.clientX, y: event.clientY };
            if (state.expandedRows.has(row.row_id)) {
              state.expandedRows.delete(row.row_id);
            } else {
              state.expandedRows.add(row.row_id);
            }
            layout();
            updateTooltip();
            scheduleDraw();
          }
          return;
        }
        state.cursorTime = xToTime(canvasX);
        state.selectedItem = null;
        if (rowIdx >= 0 && rowIdx < state.rows.length) {
          const row = state.rows[rowIdx];
          const item = findItemAt(row, rowIdx, canvasX, canvasY);
          if (item) state.selectedItem = { row, item, x: event.clientX, y: event.clientY };
        }
        updateStats();
        scheduleDraw();
      });
      canvas.addEventListener('wheel', (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        if (x < LABEL_WIDTH) {
          event.preventDefault();
          viewport.scrollTop += event.deltaY;
          return;
        }
        event.preventDefault();
        if (Math.abs(event.deltaX) > Math.abs(event.deltaY) * 0.25) {
          panTimelineByPixels(event.deltaX);
          return;
        }
        handleZoom(x, event.deltaY);
      }, { passive: false });
      window.addEventListener('resize', () => {
        layout();
        scheduleDraw();
      });
    }

    async function boot() {
      const response = await fetch(CACHE_FILE);
      const data = await response.json();
      document.getElementById('title').textContent = data.pt_path.split('/').pop() + ' trace timeline';
      state.rawRows = data.rows || [];
      const attemptValues = [...new Set(
        state.rawRows.flatMap(row => (row.items || []).map(item => item.attempt)).filter(v => v != null)
      )].sort((a, b) => a - b);
      document.getElementById('attemptValue').innerHTML =
        ['<option value="all">all attempt segments</option>'].concat(
          attemptValues.map(value => `<option value="${value}">attempt ${value} segments</option>`)
        ).join('');
      state.globalStart = data.global_start;
      state.globalEnd = data.global_end;
      state.cursorTime = state.globalStart;
      fitToRows(state.rawRows);
      attachEvents();
      applyFilterAndSort();
    }

    boot().catch((error) => {
      document.body.innerHTML = `<pre style="padding:16px">${String(error)}</pre>`;
      console.error(error);
    });
  </script>
</body>
</html>
"""


def ensure_html(paths: TimelinePaths) -> None:
    title = f"{paths.pt_path.name} trace timeline"
    html = HTML_TEMPLATE.replace("__CACHE_FILE__", paths.cache_path.name).replace("__TITLE__", title)
    with paths.html_path.open("w", encoding="utf-8") as handle:
        handle.write(html)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_directory(directory: Path, port: int) -> None:
    handler = functools.partial(QuietHandler, directory=str(directory))
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Serving http://127.0.0.1:{port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt_path", help="Path to rollout debug dump .pt file")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild cache even if it already exists")
    parser.add_argument(
        "--serve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start a local static file server for the generated HTML",
    )
    parser.add_argument("--port", type=int, default=9999, help="Port for --serve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pt_path = Path(args.pt_path).expanduser().resolve()
    if not pt_path.exists():
        raise SystemExit(f"pt file not found: {pt_path}")

    paths = _timeline_paths(pt_path)
    cache_data = ensure_cache(paths, rebuild=args.rebuild)
    ensure_html(paths)

    print(f"pt: {paths.pt_path}")
    print(f"cache: {paths.cache_path}")
    print(f"html: {paths.html_path}")
    print(f"samples: {cache_data['sample_count']}")

    if args.serve:
        serve_directory(paths.html_path.parent, args.port)


if __name__ == "__main__":
    main()
