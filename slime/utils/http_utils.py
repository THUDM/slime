import asyncio
import multiprocessing
import os
import random
import socket
from typing import Optional

import httpx

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
        except socket.error:
            return False
        except OverflowError:
            return False


def get_host_info():
    hostname = socket.gethostname()

    if env_overwrite_local_ip := os.getenv(SLIME_HOST_IP_ENV, None):
        local_ip = env_overwrite_local_ip
    else:
        try:
            local_ip = socket.gethostbyname(hostname)
        except socket.gaierror:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
                udp_sock.connect(("8.8.8.8", 80))  # Google DNS
                local_ip = udp_sock.getsockname()[0]

    return hostname, local_ip


def run_router(args):
    try:
        from sglang_router.launch_router import launch_router

        router = launch_router(args)
        if router is None:
            return 1
        return 0
    except Exception as e:
        print(e)
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


_http_client: Optional[httpx.AsyncClient] = None
_client_concurrency: int = 0

# Optional Ray-based distributed POST dispatch
_distributed_post_enabled: bool = False
_post_actors = []  # type: List[object]
_post_actor_idx: int = 0


def _next_actor():
    global _post_actor_idx
    if not _post_actors:
        return None
    actor = _post_actors[_post_actor_idx % len(_post_actors)]
    _post_actor_idx = (_post_actor_idx + 1) % len(_post_actors)
    return actor


async def _post(client, url, payload, max_retries=60):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = await client.post(url, json=payload or {})
            response.raise_for_status()
            try:
                output = response.json()
            except:
                output = response.text
        except Exception as e:
            retry_count += 1
            print(f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url})")
            if retry_count >= max_retries:
                print(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            await asyncio.sleep(1)
            continue
        break

    return output


def init_http_client(args):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    global _http_client, _client_concurrency, _distributed_post_enabled
    if not args.rollout_num_gpus:
        return

    _client_concurrency = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=_client_concurrency),
            timeout=httpx.Timeout(None),
        )

    # Optionally initialize distributed POST via Ray without changing interfaces
    if args.use_distributed_post:
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True


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
            )

        async def do_post(self, url, payload, max_retries=60):
            return await _post(self._client, url, payload, max_retries)

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


async def post(url, payload, max_retries=60):
    # If distributed mode is enabled and actors exist, dispatch via Ray.
    if _distributed_post_enabled and _post_actors:
        try:
            import ray

            actor = _next_actor()
            if actor is not None:
                # Use a thread to avoid blocking the event loop on ray.get
                obj_ref = actor.do_post.remote(url, payload, max_retries)
                return await asyncio.to_thread(ray.get, obj_ref)
        except Exception as e:
            print(f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})")
            # fall through to local

    return await _post(_http_client, url, payload, max_retries)


async def get(url):
    response = await _http_client.get(url)
    response.raise_for_status()
    output = response.json()
    return output
