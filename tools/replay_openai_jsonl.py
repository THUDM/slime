from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_INPUT_FILE = Path(__file__).resolve().parent.parent / "tob-glm-5.filtered.jsonl"
STREAM_MODE_CHOICES = ("inherit", "true", "false")


@dataclass
class ReplayStats:
    submitted: int = 0
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    stream_requests: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, result: dict[str, Any]) -> None:
        self.completed += 1
        if result["ok"]:
            self.succeeded += 1
        else:
            self.failed += 1

        if result.get("stream"):
            self.stream_requests += 1

        status_key = str(result.get("status_code", "transport_error"))
        self.status_counts[status_key] = self.status_counts.get(status_key, 0) + 1

        latency_ms = result.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            self.latencies_ms.append(float(latency_ms))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay OpenAI-compatible chat completion payloads from a JSONL file against an SGLang router.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/replay_openai_jsonl.py --port 30000 --concurrency 32\n"
            "  python tools/replay_openai_jsonl.py --port 30000 --replay-times 1\n"
            "  python tools/replay_openai_jsonl.py --base-url http://127.0.0.1:30000/v1 --concurrency 64 --max-requests 100\n"
            "  python tools/replay_openai_jsonl.py --port 30000 --model custom --stream-mode false --output-file /tmp/replay.jsonl"
        ),
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE, help="Input JSONL file to replay")
    parser.add_argument("--host", default="127.0.0.1", help="Router host when --base-url is not provided")
    parser.add_argument("--port", type=int, help="Router port when --base-url is not provided")
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:30000/v1. If omitted, it is built from --host/--port.",
    )
    parser.add_argument("--endpoint", default="/chat/completions", help="Endpoint path to append to the base URL")
    parser.add_argument("--api-key", help="Optional Bearer token for the OpenAI-compatible endpoint")
    parser.add_argument("--concurrency", type=int, default=32, help="Maximum number of in-flight requests")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=0, help="Retry count for 429/5xx/transport failures")
    parser.add_argument("--max-requests", type=int, help="Stop after replaying at most this many input lines")
    parser.add_argument("--replay-times", type=int, default=10, help="Replay the input dataset this many times")
    parser.add_argument("--model", help="Override model for every request")
    parser.add_argument("--default-model", default="custom", help="Fallback model when the payload has no model")
    parser.add_argument(
        "--stream-mode",
        choices=STREAM_MODE_CHOICES,
        default="inherit",
        help="Whether to keep the payload stream flag or override it",
    )
    parser.add_argument("--output-file", type=Path, help="Optional JSONL file for per-request results")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --output-file if it already exists")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print one progress line every N completed requests; set <= 0 to disable",
    )
    args = parser.parse_args()

    if not args.base_url and args.port is None:
        parser.error("Either --base-url or --port must be provided.")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    if args.retries < 0:
        parser.error("--retries cannot be negative.")
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive when provided.")
    if args.replay_times <= 0:
        parser.error("--replay-times must be positive.")

    return args


def build_request_url(args: argparse.Namespace) -> str:
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    if args.base_url:
        base_url = args.base_url.rstrip("/")
    else:
        base_url = f"http://{args.host}:{args.port}/v1"
    if base_url.endswith(endpoint):
        return base_url
    return f"{base_url}{endpoint}"


def build_headers(api_key: str | None, stream: bool) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def normalize_payload(raw_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(raw_payload)

    if args.model is not None:
        payload["model"] = args.model
    elif not str(payload.get("model", "") or "").strip():
        payload["model"] = args.default_model

    if args.stream_mode != "inherit":
        payload["stream"] = args.stream_mode == "true"
        if not payload["stream"]:
            payload.pop("stream_options", None)

    return payload


def decode_body(content: bytes) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content.decode("utf-8", errors="replace")


def extract_delta_text(delta: Any, key: str) -> list[str]:
    if not isinstance(delta, dict):
        return []

    value = delta.get(key)
    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        text_parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return text_parts

    return []


async def dispatch_request(
    client: httpx.AsyncClient,
    request_url: str,
    api_key: str | None,
    payload: dict[str, Any],
) -> tuple[int, str | None, Any]:
    stream = bool(payload.get("stream", False))
    headers = build_headers(api_key, stream=stream)

    if not stream:
        response = await client.post(request_url, json=payload, headers=headers)
        body = decode_body(response.content)
        return response.status_code, response.headers.get("x-request-id"), body

    async with client.stream("POST", request_url, json=payload, headers=headers) as response:
        request_id = response.headers.get("x-request-id")
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            body = decode_body(await response.aread())
            return response.status_code, request_id, body

        content_parts = []
        reasoning_parts = []
        finish_reasons = []
        tool_call_deltas = []
        usage = None
        chunk_count = 0
        raw_events = 0

        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                continue

            chunk_count += 1
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                raw_events += 1
                continue

            event_usage = event.get("usage")
            if event_usage is not None:
                usage = event_usage

            for choice in event.get("choices", []):
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    finish_reasons.append(finish_reason)

                delta = choice.get("delta", {})
                content_parts.extend(extract_delta_text(delta, "content"))
                reasoning_parts.extend(extract_delta_text(delta, "reasoning"))
                reasoning_parts.extend(extract_delta_text(delta, "reasoning_content"))

                if delta.get("tool_calls") is not None:
                    tool_call_deltas.append(delta["tool_calls"])

        body = {
            "mode": "stream",
            "content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts) or None,
            "finish_reasons": finish_reasons or None,
            "usage": usage,
            "chunk_count": chunk_count,
        }
        if tool_call_deltas:
            body["tool_call_deltas"] = tool_call_deltas
        if raw_events:
            body["raw_event_count"] = raw_events
        return response.status_code, request_id, body


def should_retry_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * ratio
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]

    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (index - lower)


def make_summary(
    stats: ReplayStats,
    elapsed_s: float,
    request_url: str,
    output_file: Path | None,
    replay_times: int,
) -> dict[str, Any]:
    throughput = stats.completed / elapsed_s if elapsed_s > 0 else None
    return {
        "request_url": request_url,
        "replay_times": replay_times,
        "submitted": stats.submitted,
        "completed": stats.completed,
        "succeeded": stats.succeeded,
        "failed": stats.failed,
        "stream_requests": stats.stream_requests,
        "non_stream_requests": stats.completed - stats.stream_requests,
        "status_counts": stats.status_counts,
        "elapsed_s": round(elapsed_s, 3),
        "throughput_rps": round(throughput, 3) if throughput is not None else None,
        "latency_ms": {
            "p50": percentile(stats.latencies_ms, 0.50),
            "p95": percentile(stats.latencies_ms, 0.95),
            "p99": percentile(stats.latencies_ms, 0.99),
        },
        "output_file": str(output_file) if output_file else None,
    }


def print_progress(stats: ReplayStats) -> None:
    print(
        f"[progress] completed={stats.completed} succeeded={stats.succeeded} failed={stats.failed} "
        f"stream={stats.stream_requests}"
    )


def emit_result(result: dict[str, Any], output_handle) -> None:
    if output_handle is None:
        return
    output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def validate_output_path(output_file: Path | None, overwrite: bool) -> None:
    if output_file is None:
        return
    if output_file.exists() and not overwrite:
        raise SystemExit(f"Output file already exists: {output_file}. Pass --overwrite to replace it.")
    output_file.parent.mkdir(parents=True, exist_ok=True)


def make_input_error_result(replay_round: int, line_number: int, error: str) -> dict[str, Any]:
    return {
        "replay_round": replay_round,
        "line_number": line_number,
        "ok": False,
        "status_code": None,
        "request_id": None,
        "latency_ms": 0.0,
        "stream": None,
        "model": None,
        "attempts": 0,
        "error": error,
        "response": None,
    }


async def send_request(
    client: httpx.AsyncClient,
    request_url: str,
    api_key: str | None,
    replay_round: int,
    line_number: int,
    payload: dict[str, Any],
    retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    last_status_code = None
    last_request_id = None
    last_response = None
    last_error = None

    for attempt in range(1, retries + 2):
        try:
            status_code, request_id, response_body = await dispatch_request(client, request_url, api_key, payload)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= retries:
                await asyncio.sleep(min(0.5 * attempt, 2.0))
                continue
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            return {
                "replay_round": replay_round,
                "line_number": line_number,
                "ok": False,
                "status_code": None,
                "request_id": None,
                "latency_ms": latency_ms,
                "stream": bool(payload.get("stream", False)),
                "model": payload.get("model"),
                "attempts": attempt,
                "error": last_error,
                "response": None,
            }

        last_status_code = status_code
        last_request_id = request_id
        last_response = response_body

        if status_code < 400:
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            return {
                "replay_round": replay_round,
                "line_number": line_number,
                "ok": True,
                "status_code": status_code,
                "request_id": request_id,
                "latency_ms": latency_ms,
                "stream": bool(payload.get("stream", False)),
                "model": payload.get("model"),
                "attempts": attempt,
                "error": None,
                "response": response_body,
            }

        last_error = f"HTTP {status_code}"
        if attempt <= retries and should_retry_status(status_code):
            await asyncio.sleep(min(0.5 * attempt, 2.0))
            continue

        break

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "replay_round": replay_round,
        "line_number": line_number,
        "ok": False,
        "status_code": last_status_code,
        "request_id": last_request_id,
        "latency_ms": latency_ms,
        "stream": bool(payload.get("stream", False)),
        "model": payload.get("model"),
        "attempts": retries + 1,
        "error": last_error,
        "response": last_response,
    }


async def collect_completed(
    pending: set[asyncio.Task],
    stats: ReplayStats,
    output_handle,
    progress_every: int,
) -> set[asyncio.Task]:
    done, still_pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        result = await task
        emit_result(result, output_handle)
        stats.record(result)
        if progress_every > 0 and stats.completed % progress_every == 0:
            print_progress(stats)
    return still_pending


async def run(args: argparse.Namespace) -> int:
    validate_output_path(args.output_file, overwrite=args.overwrite)
    request_url = build_request_url(args)
    input_file = args.input_file.resolve()
    if not input_file.exists():
        raise SystemExit(f"Input file does not exist: {input_file}")

    print(
        f"Replaying {input_file} -> {request_url} with concurrency={args.concurrency}, replay_times={args.replay_times}"
    )

    stats = ReplayStats()
    pending: set[asyncio.Task] = set()
    output_handle = args.output_file.open("w", encoding="utf-8") if args.output_file else None
    started = time.perf_counter()

    try:
        limits = httpx.Limits(
            max_connections=args.concurrency,
            max_keepalive_connections=max(20, args.concurrency),
        )
        timeout = httpx.Timeout(args.timeout)
        async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=True, trust_env=False) as client:
            with input_file.open("r", encoding="utf-8") as input_handle:
                for replay_round in range(1, args.replay_times + 1):
                    input_handle.seek(0)
                    for line_number, line in enumerate(input_handle, start=1):
                        if args.max_requests is not None and stats.submitted >= args.max_requests:
                            break

                        stripped = line.strip()
                        if not stripped:
                            continue

                        try:
                            raw_payload = json.loads(stripped)
                        except json.JSONDecodeError as exc:
                            stats.submitted += 1
                            result = make_input_error_result(replay_round, line_number, f"Invalid JSON: {exc}")
                            emit_result(result, output_handle)
                            stats.record(result)
                            continue

                        if not isinstance(raw_payload, dict):
                            stats.submitted += 1
                            result = make_input_error_result(replay_round, line_number, "JSON line must be an object")
                            emit_result(result, output_handle)
                            stats.record(result)
                            continue

                        payload = normalize_payload(raw_payload, args)
                        pending.add(
                            asyncio.create_task(
                                send_request(
                                    client=client,
                                    request_url=request_url,
                                    api_key=args.api_key,
                                    replay_round=replay_round,
                                    line_number=line_number,
                                    payload=payload,
                                    retries=args.retries,
                                )
                            )
                        )
                        stats.submitted += 1

                        if len(pending) >= args.concurrency:
                            pending = await collect_completed(
                                pending,
                                stats=stats,
                                output_handle=output_handle,
                                progress_every=args.progress_every,
                            )

                    if args.max_requests is not None and stats.submitted >= args.max_requests:
                        break

            while pending:
                pending = await collect_completed(
                    pending,
                    stats=stats,
                    output_handle=output_handle,
                    progress_every=args.progress_every,
                )
    finally:
        if output_handle is not None:
            output_handle.close()

    summary = make_summary(
        stats,
        elapsed_s=time.perf_counter() - started,
        request_url=request_url,
        output_file=args.output_file,
        replay_times=args.replay_times,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if stats.failed == 0 else 1


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
