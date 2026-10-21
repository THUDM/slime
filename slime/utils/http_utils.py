import asyncio
import ipaddress
import json
import logging
import os
import random
import socket
import ssl

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


_http_client: httpx.AsyncClient | None = None
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


class _ShardedHTTPTransport(httpx.AsyncBaseTransport):
    """Reduce asyncio CPU stalls from httpcore's per-request connection scans.

    Split this client's connection budget across pools of at most 128
    connections, so each synchronous scan touches a small pool. The pools
    operate concurrently: 128 is a per-pool limit, not a client-wide request
    limit. Their connection limits sum to max_connections; requests assigned
    to a full pool wait for a connection there.

    Allow each pool to retain its entire connection budget as idle keep-alive
    connections, subject to the usual expiry. This avoids closing/reopening
    most sockets after a request wave; httpx otherwise keeps at most 20 idle
    connections per pool. That idle limit is separate from active concurrency.
    """

    def __init__(self, max_connections: int):
        count = max(1, (max_connections + 127) // 128)
        size, remainder = divmod(max(1, max_connections), count)
        tls = ssl.create_default_context()
        self._transports = [
            httpx.AsyncHTTPTransport(
                verify=tls,
                trust_env=False,
                limits=httpx.Limits(
                    max_connections=size + (i < remainder),
                    max_keepalive_connections=size + (i < remainder),
                ),
            )
            for i in range(count)
        ]
        self._next = 0

    async def handle_async_request(self, request):
        transport = self._transports[self._next]
        self._next = (self._next + 1) % len(self._transports)
        return await transport.handle_async_request(request)

    async def aclose(self):
        await asyncio.gather(*(transport.aclose() for transport in self._transports))


async def _post(client, url, payload, max_retries=60, headers=None, timeout=None, score_centering_top_k=None):
    retry_count = 0
    while retry_count < max_retries:
        response = None
        try:
            request_kwargs = {"json": payload or {}, "headers": headers}
            if timeout is not None:
                request_kwargs["timeout"] = timeout
            response = await client.post(url, **request_kwargs)
            response.raise_for_status()
            content = await response.aread()
            try:
                if score_centering_top_k is not None:
                    from slime.utils.score_centering import decode_score_centering_response

                    output = await asyncio.to_thread(decode_score_centering_response, content, score_centering_top_k)
                else:
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
                f"Error: {e!r}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
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


def get_rollout_num_engines(args) -> int:
    """Return the number of rollout HTTP engines behind the router."""
    if (num_engines := getattr(args, "rollout_num_engines", None)) is not None:
        return int(num_engines)

    rollout_num_gpus = getattr(args, "rollout_num_gpus", None) or 0
    rollout_num_gpus_per_engine = getattr(args, "rollout_num_gpus_per_engine", None) or 1
    if rollout_num_gpus <= 0:
        return 0
    return max(1, rollout_num_gpus // rollout_num_gpus_per_engine)


def init_http_client(args, *, concurrency: int | None = None):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    global _http_client, _client_concurrency, _distributed_post_enabled
    num_engines = get_rollout_num_engines(args)
    if num_engines <= 0:
        return

    _client_concurrency = concurrency if concurrency is not None else args.sglang_server_concurrency * num_engines
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            transport=_ShardedHTTPTransport(_client_concurrency),
            timeout=httpx.Timeout(None),
            trust_env=False,  # internal SGLang comm only — never route through system proxy
        )

    # Optionally initialize distributed POST via Ray without changing interfaces
    if args.use_distributed_post:
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True


def _init_ray_distributed_post(args):
    """Initialize one or more Ray async actors per node for HTTP POST.

    Uses NodeAffinitySchedulingStrategy to place actors on alive Ray nodes.
    Creates ``args.num_gpus_per_node`` actors per node.
    """
    global _post_actors
    if _post_actors:
        return  # Already initialized

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from slime.ray.utils import add_default_ray_env_vars

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
                transport=_ShardedHTTPTransport(concurrency),
                timeout=httpx.Timeout(None),
                trust_env=False,  # internal SGLang comm only — never route through system proxy
            )

        async def do_post(self, url, payload, max_retries=60, headers=None, timeout=None, score_centering_top_k=None):
            return await _post(
                self._client,
                url,
                payload,
                max_retries,
                headers=headers,
                timeout=timeout,
                score_centering_top_k=score_centering_top_k,
            )

    # Create actors per node
    created = []
    total_actors = max(1, len(nodes) * args.num_gpus_per_node)
    per_actor_conc = max(1, (_client_concurrency + total_actors - 1) // total_actors)

    for node in nodes:
        node_id = node["NodeID"]
        scheduling = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        for _ in range(args.num_gpus_per_node):
            actor = _HttpPosterActor.options(
                name=None,
                lifetime="detached",
                runtime_env={"env_vars": add_default_ray_env_vars()},
                scheduling_strategy=scheduling,
                max_concurrency=per_actor_conc,
                # Use tiny CPU to schedule
                num_cpus=0.001,
            ).remote(per_actor_conc)
            created.append(actor)

    _post_actors = created


async def post(url, payload, max_retries=60, headers=None, timeout=None, score_centering_top_k=None):
    # If distributed mode is enabled and actors exist, dispatch via Ray.
    if _distributed_post_enabled and _post_actors:
        try:
            actor = _next_actor()
            if actor is not None:
                # Await the Ray ObjectRef directly. The previous
                # `asyncio.to_thread(ray.get, obj_ref)` blocked an OS thread
                # from the default ThreadPoolExecutor (capped at
                # `min(32, cpu+4)`), which becomes a hard upper bound on the
                # number of in-flight POSTs that can be waited on in parallel
                # and produces large tail latencies under high concurrency.
                obj_ref = actor.do_post.remote(
                    url,
                    payload,
                    max_retries,
                    headers=headers,
                    timeout=timeout,
                    score_centering_top_k=score_centering_top_k,
                )
                return await obj_ref
        except Exception as e:
            logger.info(f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})")
            # fall through to local

    return await _post(
        _http_client,
        url,
        payload,
        max_retries,
        headers=headers,
        timeout=timeout,
        score_centering_top_k=score_centering_top_k,
    )


async def get(url, *, timeout=None):
    response = await _http_client.get(url, timeout=timeout)
    try:
        response.raise_for_status()
        content = await response.aread()
        return json.loads(content)
    finally:
        await response.aclose()
