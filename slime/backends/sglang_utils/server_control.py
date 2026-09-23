import asyncio
import logging
from typing import Any

from slime.utils.http_utils import get, post

logger = logging.getLogger(__name__)

ABORT_RETRY_INTERVAL_SECONDS = 3
DEFAULT_ABORT_TIMEOUT_SECONDS = 180.0
DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS = 10.0


def num_requests_from_load(load: Any) -> int:
    if isinstance(load, list):
        return sum(num_requests_from_load(item) for item in load)

    if not isinstance(load, dict):
        return 0

    if "loads" in load:
        return num_requests_from_load(load["loads"])

    for key in ("num_reqs", "num_total_reqs", "total_reqs"):
        value = load.get(key)
        if isinstance(value, int):
            core_requests = value
            break
    else:
        running = load.get("num_running_reqs", load.get("total_running_reqs"))
        waiting = load.get("num_waiting_reqs", load.get("total_waiting_reqs"))
        core_requests = (running if isinstance(running, int) else 0) + (waiting if isinstance(waiting, int) else 0)

    # The core count can be zero while disaggregated prefill transfers are
    # still alive. Inspect all detailed counters; max avoids double-counting
    # requests represented in more than one view while preserving idle checks.
    detailed_counts = [core_requests]
    live_request_keys = {
        "num_reqs",
        "num_running_reqs",
        "num_waiting_reqs",
        "num_total_reqs",
        "total_reqs",
        "total_running_reqs",
        "total_waiting_reqs",
        "waiting",
        "grammar",
    }
    for key, value in load.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and (key in live_request_keys or key.endswith("_queue_reqs")):
            detailed_counts.append(value)
        elif key in {"disaggregation", "disagg", "queues", "inflight"}:
            detailed_counts.append(num_requests_from_load(value))
    return max(detailed_counts, default=0)


def non_idle_request_details(load: Any) -> list[dict[str, Any]]:
    """Return a compact view of non-empty SGLang queues."""
    if not isinstance(load, dict):
        return []
    loads = load.get("loads", [load])
    if not isinstance(loads, list):
        return []

    details: list[dict[str, Any]] = []
    for rank_load in loads:
        if not isinstance(rank_load, dict):
            continue
        dp_rank = rank_load.get("dp_rank")
        rank_has_detail = False
        inflight = rank_load.get("inflight")
        if isinstance(inflight, list):
            for queue in inflight:
                if not isinstance(queue, dict):
                    continue
                reqs = queue.get("reqs") if isinstance(queue.get("reqs"), list) else []
                count = queue.get("num_reqs", len(reqs))
                if isinstance(count, int) and count > 0:
                    rank_has_detail = True
                    details.append(
                        {"dp_rank": dp_rank, "queue": queue.get("name", "unknown"), "num_reqs": count, "reqs": reqs}
                    )
        if rank_has_detail:
            continue

        counters: dict[str, int] = {}
        for section_name in ("disaggregation", "queues"):
            section = rank_load.get(section_name)
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    counters[f"{section_name}.{key}"] = value
        for key in ("num_running_reqs", "num_waiting_reqs", "num_total_reqs"):
            value = rank_load.get(key)
            if isinstance(value, int) and value > 0:
                counters[key] = value
        if counters:
            details.append({"dp_rank": dp_rank, "counters": counters})
    return details


async def _abort_server_once(url: str, request_timeout: float) -> None:
    await asyncio.wait_for(
        post(f"{url}/abort_request", {"abort_all": True}, max_retries=1, timeout=request_timeout),
        timeout=request_timeout,
    )


async def _get_server_load(url: str, request_timeout: float) -> Any:
    return await asyncio.wait_for(
        get(f"{url}/v1/loads?include=core,disagg,queues,inflight", timeout=request_timeout),
        timeout=request_timeout,
    )


async def abort_server_until_idle(
    url: str,
    retry_interval: float = ABORT_RETRY_INTERVAL_SECONDS,
    *,
    timeout: float = DEFAULT_ABORT_TIMEOUT_SECONDS,
    request_timeout: float = DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS,
) -> None:
    if timeout <= 0 or request_timeout <= 0:
        raise ValueError("abort and control-request timeouts must be positive")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    attempt = 1
    last_error: Exception | None = None
    last_num_requests: int | None = None
    last_load: Any = None
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            detail = (
                f"last observed request count={last_num_requests}"
                if last_num_requests is not None
                else f"last error={last_error!r}"
            )
            if last_load is not None:
                detail += f", non-idle queues={non_idle_request_details(last_load)!r}"
            raise TimeoutError(f"Timed out draining SGLang server {url} after {timeout:.1f}s ({detail})")

        logger.info(f"Abort request for SGLang server {url}")
        per_request_timeout = min(request_timeout, remaining)
        try:
            await _abort_server_once(url, per_request_timeout)
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to abort SGLang server at {url}: {e}")

        try:
            remaining = deadline - loop.time()
            if remaining <= 0:
                continue
            load = await _get_server_load(url, min(request_timeout, remaining))
            num_requests = num_requests_from_load(load)
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to get SGLang server load from {url}: {e}")
        else:
            last_load = load
            last_num_requests = num_requests
            if num_requests <= 0:
                return
            logger.info(
                "SGLang server %s still has %d requests after abort attempt %d; "
                "non-idle queues=%r; retrying in %s seconds.",
                url,
                num_requests,
                attempt,
                non_idle_request_details(load),
                retry_interval,
            )

        remaining = deadline - loop.time()
        if remaining > 0:
            await asyncio.sleep(min(retry_interval, remaining))
        attempt += 1


async def abort_servers_until_idle(
    urls: list[str],
    *,
    timeout: float = DEFAULT_ABORT_TIMEOUT_SECONDS,
    request_timeout: float = DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS,
) -> None:
    results = await asyncio.gather(
        *(abort_server_until_idle(url, timeout=timeout, request_timeout=request_timeout) for url in urls),
        return_exceptions=True,
    )
    failures = [(url, result) for url, result in zip(urls, results, strict=True) if isinstance(result, BaseException)]
    if failures:
        detail = "; ".join(f"{url}: {error}" for url, error in failures)
        raise RuntimeError(f"Failed to abort all SGLang servers to a confirmed idle state: {detail}") from failures[0][
            1
        ]
