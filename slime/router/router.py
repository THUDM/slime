import argparse
import asyncio
import json
import logging
import uuid
from enum import Enum
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response


logger = logging.getLogger(__name__)


class WorkerType(str, Enum):
    REGULAR = "regular"
    PREFILL = "prefill"
    DECODE = "decode"


class WorkerInfo:
    """Metadata for a registered worker."""

    __slots__ = ("url", "worker_type", "active_requests", "consecutive_failures", "bootstrap_port")

    def __init__(self, url: str, worker_type: WorkerType = WorkerType.REGULAR, bootstrap_port: int | None = None):
        self.url = url
        self.worker_type = worker_type
        self.active_requests: int = 0
        self.consecutive_failures: int = 0
        self.bootstrap_port = bootstrap_port


def run_router(args):
    """Run the Slime router with the specified configuration."""
    slime_router = SlimeRouter(args, verbose=False)
    uvicorn.run(slime_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class SlimeRouter:
    def __init__(self, args, verbose=False):
        """Initialize the slime-router."""
        self.args = args
        self.verbose = verbose

        self.app = FastAPI()
        self.app.add_event_handler("startup", self._start_background_health_check)

        # URL -> WorkerInfo
        self.workers: dict[str, WorkerInfo] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()
        self.max_weight_version = None

        # --- Connection pool ---
        max_connections = getattr(args, "slime_router_max_connections", None)
        if max_connections is None:
            max_connections = (
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )
        # Generous keep-alive pool for high concurrency
        max_keepalive = max(max_connections // 2, 20)

        timeout = getattr(args, "slime_router_timeout", None)

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive,
                keepalive_expiry=30,
            ),
            timeout=httpx.Timeout(timeout),
            http2=True,
        )

        self._setup_routes()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _setup_routes(self):
        """Setup all the HTTP routes."""
        self.app.post("/add_worker")(self.add_worker)
        self.app.post("/remove_worker")(self.remove_worker)
        self.app.post("/workers")(self.add_worker_v2)
        self.app.get("/workers")(self.list_workers_v2)
        self.app.get("/list_workers")(self.list_workers)
        self.app.get("/health")(self.health)
        # Catch-all route for proxying — must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    # ------------------------------------------------------------------
    # Health check background loop
    # ------------------------------------------------------------------

    async def _start_background_health_check(self):
        asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url: str):
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return url, True
            logger.debug(f"[slime-router] Worker {url} is unhealthy (Status: {response.status_code})")
        except Exception as e:
            logger.debug(f"[slime-router] Worker {url} health check failed: {e}")
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        interval = self.args.rollout_health_check_interval
        threshold = self.args.slime_router_health_check_failure_threshold

        while True:
            try:
                await asyncio.sleep(interval)

                urls = [u for u in self.workers if u not in self.dead_workers]
                if not urls:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                for url, is_healthy in results:
                    if url not in self.workers:
                        continue
                    if not is_healthy:
                        self.workers[url].consecutive_failures += 1
                        if self.workers[url].consecutive_failures >= threshold:
                            logger.warning(
                                f"[slime-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                    else:
                        self.workers[url].consecutive_failures = 0

                alive = sum(1 for u in self.workers if u not in self.dead_workers)
                logger.debug(f"[slime-router] Health check complete. {alive} workers healthy.")

            except asyncio.CancelledError:
                logger.warning("[slime-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[slime-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Worker selection
    # ------------------------------------------------------------------

    def _healthy_workers(self, worker_type: WorkerType | None = None) -> list[WorkerInfo]:
        """Return live workers, optionally filtered by type."""
        workers = [w for url, w in self.workers.items() if url not in self.dead_workers]
        if worker_type is not None:
            workers = [w for w in workers if w.worker_type == worker_type]
        return workers

    def _select_by_least_inflight(self, candidates: list[WorkerInfo]) -> WorkerInfo:
        """Pick the worker with the fewest active requests."""
        if not candidates:
            raise RuntimeError("No healthy workers available in the pool")
        return min(candidates, key=lambda w: w.active_requests)

    def _is_pd_mode(self) -> bool:
        """Check if PD disaggregation is active (prefill workers exist)."""
        return any(
            w.worker_type == WorkerType.PREFILL for url, w in self.workers.items() if url not in self.dead_workers
        )

    def _pick_pd_pair(self) -> tuple[WorkerInfo, WorkerInfo]:
        """Pick a (prefill, decode) worker pair using least-inflight."""
        prefill_candidates = self._healthy_workers(WorkerType.PREFILL)
        decode_candidates = self._healthy_workers(WorkerType.DECODE)
        if not prefill_candidates:
            raise RuntimeError("No healthy prefill workers available")
        if not decode_candidates:
            raise RuntimeError("No healthy decode workers available")
        prefill = self._select_by_least_inflight(prefill_candidates)
        decode = self._select_by_least_inflight(decode_candidates)
        prefill.active_requests += 1
        decode.active_requests += 1
        return prefill, decode

    def _pick_worker(self) -> WorkerInfo:
        """Pick a single worker via least-inflight (non-PD mode)."""
        candidates = self._healthy_workers()
        worker = self._select_by_least_inflight(candidates)
        worker.active_requests += 1
        return worker

    def _finish_worker(self, worker: WorkerInfo):
        """Mark the request to the given worker as finished."""
        worker.active_requests -= 1
        assert worker.active_requests >= 0, f"Worker {worker.url} active_requests went negative"

    # ------------------------------------------------------------------
    # Proxy (streaming)
    # ------------------------------------------------------------------

    async def proxy(self, request: Request, path: str):
        """Stream-proxy requests to a selected backend worker.

        In PD disaggregation mode, picks a (prefill, decode) pair, injects
        bootstrap info, and sends the same request to both workers concurrently
        (mirroring sgl-model-gateway behaviour).  The decode worker's response
        is returned to the caller.
        """
        body = await request.body()
        headers = dict(request.headers)

        if self._is_pd_mode():
            return await self._proxy_pd(path, body, headers)
        else:
            worker = self._pick_worker()
            try:
                return await self._forward_to_worker(worker, path, body, headers)
            finally:
                self._finish_worker(worker)

    # --- PD dual-dispatch helpers ---

    def _bootstrap_host_from_url(self, worker_url: str) -> str:
        """Extract the hostname from a worker URL for bootstrap."""
        return urlparse(worker_url).hostname or "127.0.0.1"

    def _inject_bootstrap(self, body: bytes, prefill: WorkerInfo) -> bytes:
        """Inject bootstrap_host / bootstrap_port / bootstrap_room into the request body."""
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            return body
        payload["bootstrap_host"] = self._bootstrap_host_from_url(prefill.url)
        payload["bootstrap_port"] = prefill.bootstrap_port
        payload["bootstrap_room"] = uuid.uuid4().hex
        return json.dumps(payload).encode()

    async def _proxy_pd(self, path: str, body: bytes, headers: dict) -> Response:
        """PD dual dispatch: send the same request to prefill + decode concurrently."""
        prefill, decode = self._pick_pd_pair()
        try:
            modified_body = self._inject_bootstrap(body, prefill)

            prefill_url = f"{prefill.url}/{path}"
            decode_url = f"{decode.url}/{path}"

            prefill_req = self.client.build_request("POST", prefill_url, content=modified_body, headers=headers)
            decode_req = self.client.build_request("POST", decode_url, content=modified_body, headers=headers)

            # Fire both concurrently; we only care about the decode response.
            _prefill_task = asyncio.ensure_future(self.client.send(prefill_req, stream=True))
            decode_response = await self.client.send(decode_req, stream=True)

            return await self._build_response(decode_response)
        finally:
            self._finish_worker(prefill)
            self._finish_worker(decode)

    async def _forward_to_worker(self, worker: WorkerInfo, path: str, body: bytes, headers: dict) -> Response:
        """Forward a request to a single worker and return its response."""
        url = f"{worker.url}/{path}"
        req = self.client.build_request("POST", url, content=body, headers=headers)
        response = await self.client.send(req, stream=True)
        return await self._build_response(response)

    async def _build_response(self, response: httpx.Response) -> Response:
        """Convert an httpx streaming response into a FastAPI response."""
        content_type = response.headers.get("content-type", "")

        if "text/event-stream" not in content_type:
            content = await response.aread()
            await response.aclose()
            try:
                data = json.loads(content)
                return JSONResponse(content=data, status_code=response.status_code)
            except Exception:
                return Response(content=content, status_code=response.status_code, media_type=content_type or None)

        async def _stream():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(_stream(), status_code=response.status_code, media_type=content_type)

    # ------------------------------------------------------------------
    # Worker management endpoints
    # ------------------------------------------------------------------

    async def add_worker(self, request: Request):
        """Add a new worker (v1 compat — query string or JSON body).

        Examples:
          POST /add_worker?url=http://127.0.0.1:10090
          POST /add_worker?url=http://127.0.0.1:10090&worker_type=prefill
          POST /add_worker  {"url": "...", "worker_type": "prefill"}
        """
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")
        worker_type_str = request.query_params.get("worker_type", "regular")

        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")
            worker_type_str = payload.get("worker_type", worker_type_str)

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "url is required (use query ?url=... or JSON body)"}
            )

        try:
            worker_type = WorkerType(worker_type_str)
        except ValueError:
            worker_type = WorkerType.REGULAR

        if worker_url not in self.workers:
            self.workers[worker_url] = WorkerInfo(url=worker_url, worker_type=worker_type)
            if self.verbose:
                print(f"[slime-router] Added new worker: {worker_url} (type={worker_type.value})")

        return {"status": "success", "worker_urls": {u: w.active_requests for u, w in self.workers.items()}}

    async def add_worker_v2(self, request: Request):
        """Add worker — SGLang Model Gateway compatible ``POST /workers`` endpoint.

        Body: {"url": "...", "worker_type": "prefill"|"decode"|"regular", "bootstrap_port": 12345}
        """
        body = await request.body()
        payload = json.loads(body) if body else {}
        worker_url = payload.get("url")
        worker_type_str = payload.get("worker_type", "regular")
        bootstrap_port = payload.get("bootstrap_port")

        if not worker_url:
            return JSONResponse(status_code=400, content={"error": "url is required in JSON body"})

        try:
            worker_type = WorkerType(worker_type_str)
        except ValueError:
            worker_type = WorkerType.REGULAR

        if worker_url not in self.workers:
            self.workers[worker_url] = WorkerInfo(
                url=worker_url, worker_type=worker_type, bootstrap_port=bootstrap_port
            )
            if self.verbose:
                print(f"[slime-router] Added new worker: {worker_url} (type={worker_type.value})")

        return {"status": "success"}

    async def remove_worker(self, request: Request):
        """Remove a worker from the pool."""
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")

        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(status_code=400, content={"error": "url is required"})

        self.workers.pop(worker_url, None)
        self.dead_workers.discard(worker_url)
        return {"status": "success"}

    async def list_workers(self, request: Request):
        """List all registered workers (v1 compat)."""
        return {"urls": list(self.workers.keys())}

    async def list_workers_v2(self, request: Request):
        """List workers — SGLang Model Gateway compatible ``GET /workers``."""
        workers_list = []
        for url, w in self.workers.items():
            entry = {
                "url": url,
                "worker_type": w.worker_type.value,
                "active_requests": w.active_requests,
                "is_healthy": url not in self.dead_workers,
            }
            if w.bootstrap_port is not None:
                entry["bootstrap_port"] = w.bootstrap_port
            workers_list.append(entry)
        return {"workers": workers_list}

    async def health(self, request: Request):
        """Router health check endpoint."""
        alive = sum(1 for u in self.workers if u not in self.dead_workers)
        return {"status": "ok", "healthy_workers": alive, "total_workers": len(self.workers)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    run_router(args)
