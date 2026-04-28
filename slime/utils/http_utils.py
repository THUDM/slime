import asyncio
import ipaddress
import json
import logging
import multiprocessing
import os
import random
import socket
import weakref

import httpx

logger = logging.getLogger(__name__)

SLIME_HOST_IP_ENV = "SLIME_HOST_IP"


def find_available_port(base_port: int):
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        if port < 60000:
            port += 42
        else:
            port -= 43


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def get_host_info():
    hostname = socket.gethostname()

    if env_overwrite_local_ip := os.getenv(SLIME_HOST_IP_ENV, None):
        return hostname, env_overwrite_local_ip

    def _is_loopback(ip):
        return ip.startswith("127.") or ip == "::1"

    def _resolve_ip(family, test_target_ip):
        """
        Attempt to get the local LAN IP for the specific family (IPv4/IPv6).
        Strategy: UDP Probe (Preferred) -> Hostname Resolution (Fallback) -> None
        """

        # Strategy 1: UDP Connect Probe (Most accurate, relies on routing table)
        # Useful when the machine has a default gateway or internet access.
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                # The IP doesn't need to be reachable, but the routing table must exist.
                s.connect((test_target_ip, 80))
                ip = s.getsockname()[0]
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass  # Route unreachable or network error, move to next strategy.

        # Strategy 2: Hostname Resolution (Fallback for offline clusters)
        # Useful for offline environments where UDP connect fails but /etc/hosts is configured.
        try:
            # getaddrinfo allows specifying the family (AF_INET or AF_INET6)
            # Result format: [(family, type, proto, canonname, sockaddr), ...]
            infos = socket.getaddrinfo(hostname, None, family=family, type=socket.SOCK_STREAM)

            for info in infos:
                ip = info[4][0]  # The first element of sockaddr is the IP
                # Must filter out loopback addresses to avoid "127.0.0.1" issues
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass

        return None

    prefer_ipv6 = os.getenv("SLIME_PREFER_IPV6", "0").lower() in ("1", "true", "yes", "on")
    local_ip = None
    final_fallback = "127.0.0.1"

    if prefer_ipv6:
        # [Strict Mode] IPv6 Only
        # 1. Try UDP V6 Probe
        # 2. Try Hostname Resolution (V6)
        # If failed, fallback to V6 loopback. Never mix with V4.
        local_ip = _resolve_ip(socket.AF_INET6, "2001:4860:4860::8888")
        final_fallback = "::1"
    else:
        # [Strict Mode] IPv4 Only (Default)
        # 1. Try UDP V4 Probe
        # 2. Try Hostname Resolution (V4)
        # If failed, fallback to V4 loopback. Never mix with V6.
        local_ip = _resolve_ip(socket.AF_INET, "8.8.8.8")
        final_fallback = "127.0.0.1"

    return hostname, local_ip or final_fallback


def _wrap_ipv6(host):
    """Wrap IPv6 address in [] if needed."""
    try:
        ipaddress.IPv6Address(host.strip("[]"))
        return f"[{host.strip('[]')}]"
    except ipaddress.AddressValueError:
        return host


def run_router(args):
    try:
        from sglang_router.launch_router import launch_router

        router = launch_router(args)
        if router is None:
            return 1
        return 0
    except Exception as e:
        logger.info(e)
        return 1


def terminate_process(process: multiprocessing.Process, timeout: float = 1.0) -> None:
    """Terminate a process gracefully, with forced kill as fallback.

    Args:
        process: The process to terminate
        timeout: Seconds to wait for graceful termination before forcing kill
    """
    if not process.is_alive():
        return

    process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()


_http_client: httpx.AsyncClient | None = None
_http_client_loop_ref: weakref.ref | None = None
_client_concurrency: int = 0

# Optional Ray-based distributed POST dispatch
_distributed_post_enabled: bool = False
_post_actors: list[object] = []
_post_actor_idx: int = 0


def _next_actor():
    global _post_actor_idx
    if not _post_actors:
        return None
    actor = _post_actors[_post_actor_idx % len(_post_actors)]
    _post_actor_idx = (_post_actor_idx + 1) % len(_post_actors)
    return actor


async def _post(client, url, payload, max_retries=60, headers=None):
    retry_count = 0
    while retry_count < max_retries:
        response = None
        try:
            response = await client.post(url, json=payload or {}, headers=headers)
            response.raise_for_status()
            content = await response.aread()
            try:
                output = json.loads(content)
            except json.JSONDecodeError:
                output = content.decode() if isinstance(content, bytes) else content
        except Exception as e:
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None

            logger.info(
                f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
            )
            if retry_count >= max_retries:
                logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            await asyncio.sleep(1)
            continue
        finally:
            if response is not None:
                await response.aclose()
        break

    return output


def init_http_client(args):
    """Record HTTP-client concurrency and optionally enable distributed POST.

    The actual ``httpx.AsyncClient`` is built lazily per event loop by
    ``_get_http_client`` on first use, so Ray actors that hop loops get a
    fresh client bound to the active loop.
    """
    global _client_concurrency, _distributed_post_enabled
    if not args.rollout_num_gpus:
        return

    _client_concurrency = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    if _client_concurrency <= 0:
        # Not fatal (we fall back to 1 in _get_http_client) but almost
        # certainly a configuration bug; surface it in logs rather than
        # silently throttling every rollout to 1 in-flight request.
        logger.warning(
            "_client_concurrency computed as %d from sglang_server_concurrency=%s, "
            "rollout_num_gpus=%s, rollout_num_gpus_per_engine=%s; check your config.",
            _client_concurrency,
            args.sglang_server_concurrency,
            args.rollout_num_gpus,
            args.rollout_num_gpus_per_engine,
        )

    if args.use_distributed_post:
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True


def _get_http_client() -> httpx.AsyncClient:
    """Get or (re)create the HTTP client for the current event loop.

    ``httpx.AsyncClient`` binds its connection pool (and the internal
    asyncio locks it uses) to the loop that created it. Ray actors can
    serve calls on different loops across re-entries (e.g. rollout →
    eval), so we detect a loop change and rebuild.

    Old-client cleanup: we deliberately do *not* call ``aclose()`` on the
    old client. It was bound to a loop that may already be dead; running
    ``aclose()`` from a different loop races on the pool's internal
    asyncio locks and can leave sockets half-closed. We drop the reference
    and let GC close the underlying sockets when the client is collected
    (httpx emits a ResourceWarning; expected). A socket leak across a
    rare loop transition is cheaper than a partial-aclose data race.

    Thread/concurrency safety: safe under asyncio because there is no
    ``await`` between the loop-change check and the assignment to
    ``_http_client``, so two concurrent callers on the same loop can't
    both pass the check and race on construction. Any future refactor
    that inserts an ``await`` here must add a per-loop lock.
    """
    global _http_client, _http_client_loop_ref
    current_loop = asyncio.get_running_loop()
    stored_loop = _http_client_loop_ref() if _http_client_loop_ref is not None else None
    if _http_client is None or stored_loop is not current_loop:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max(_client_concurrency, 1)),
            timeout=httpx.Timeout(None),
            trust_env=False,  # internal SGLang comm only — never route through system proxy
        )
        _http_client_loop_ref = weakref.ref(current_loop)
    return _http_client


def _init_ray_distributed_post(args):
    """Initialize one or more Ray async actors per node for HTTP POST.

    Uses NodeAffinitySchedulingStrategy to place actors on distinct nodes.
    Controlled by SLIME_HTTP_POST_ACTORS_PER_NODE.
    """
    global _post_actors
    if _post_actors:
        return  # Already initialized

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Discover alive nodes
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise RuntimeError("No alive Ray nodes to place HTTP POST actors.")

    # Define the async actor
    @ray.remote
    class _HttpPosterActor:
        def __init__(self, concurrency: int):
            # Lazy creation to this actor's event loop
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=max(1, concurrency)),
                timeout=httpx.Timeout(None),
                trust_env=False,  # internal SGLang comm only — never route through system proxy
            )

        async def do_post(self, url, payload, max_retries=60, headers=None):
            return await _post(self._client, url, payload, max_retries, headers=headers)

    # Create actors per node
    created = []
    # Distribute client concurrency across actors (at least 1 per actor)
    per_actor_conc = (_client_concurrency + len(nodes)) // len(nodes)

    for node in nodes:
        node_id = node["NodeID"]
        scheduling = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        for _ in range(args.num_gpus_per_node):
            actor = _HttpPosterActor.options(
                name=None,
                lifetime="detached",
                scheduling_strategy=scheduling,
                max_concurrency=per_actor_conc,
                # Use tiny CPU to schedule
                num_cpus=0.001,
            ).remote(per_actor_conc)
            created.append(actor)

    _post_actors = created


async def post(url, payload, max_retries=60, headers=None):
    # If distributed mode is enabled and actors exist, dispatch via Ray.
    if _distributed_post_enabled and _post_actors:
        try:
            import ray

            actor = _next_actor()
            if actor is not None:
                # Await the Ray ObjectRef directly. The previous
                # `asyncio.to_thread(ray.get, obj_ref)` blocked an OS thread
                # from the default ThreadPoolExecutor (capped at
                # `min(32, cpu+4)`), which becomes a hard upper bound on the
                # number of in-flight POSTs that can be waited on in parallel
                # and produces large tail latencies under high concurrency.
                obj_ref = actor.do_post.remote(url, payload, max_retries, headers=headers)
                return await obj_ref
        except Exception as e:
            logger.info(f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})")
            # fall through to local

    return await _post(_get_http_client(), url, payload, max_retries, headers=headers)


async def get(url):
    response = await _get_http_client().get(url)
    response.raise_for_status()
    content = await response.aread()
    output = json.loads(content)
    return output
