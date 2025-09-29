import argparse
import asyncio
import json
import random
from typing import Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from slime.utils.misc import load_function


def run_router(args):
    """
    Run the Slime router with the specified configuration.
    """
    # Initialize the router with tokenizer and lazy worker initialization
    slime_router = SlimeRouter(args, verbose=False)

    # Start the server
    uvicorn.run(slime_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class SlimeRouter:
    def __init__(self, args, verbose=False):
        """Initialize the slime-router with SGLang router address"""
        self.args = args
        self.verbose = verbose

        self.app = FastAPI()

        # Worker information with lock protection
        self.worker_urls: dict[str, int] = {}
        self.worker_health: dict[str, str] = {}
        self.worker_lock = asyncio.Lock()
        self.max_weight_version = None

        # Health check configuration
        self.health_check_interval = getattr(args, "health_check_interval", 30)
        self.health_check_timeout = getattr(args, "health_check_timeout", 5.0)
        self.health_check_jitter = getattr(args, "health_check_jitter", 2.0)

        # TODO: remove this hardcode
        # Main client for proxying user requests
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=args.sglang_server_concurrency
                * args.rollout_num_gpus
                // args.rollout_num_gpus_per_engine
            ),
            timeout=httpx.Timeout(None),
        )

        # Dedicated, reusable client for health checks
        self.health_client = httpx.AsyncClient(timeout=httpx.Timeout(self.health_check_timeout))
        self._health_check_task: Optional[asyncio.Task] = None

        self._setup_routes()

        for middleware_path in args.slime_router_middleware_paths or []:
            if self.verbose:
                print(f"[slime-router] Loading middleware from: {middleware_path}")
            middleware = load_function(middleware_path)
            self.app.add_middleware(middleware, router=self)

        # Add startup and shutdown event handlers
        self.app.on_event("startup")(self.start_background_tasks)
        self.app.on_event("shutdown")(self.shutdown_background_tasks)

    def start_background_tasks(self):
        """Starts background tasks."""
        self._health_check_task = asyncio.create_task(self.background_health_check_task())
        if self.verbose:
            print(f"[slime-router] Started background health check task (interval: {self.health_check_interval}s)")

    async def shutdown_background_tasks(self):
        """Shuts down background tasks gracefully."""
        if self.verbose:
            print("[slime-router] Shutting down background tasks...")

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                if self.verbose:
                    print("[slime-router] Health check task cancelled")

        await self.health_client.aclose()
        await self.client.aclose()

        if self.verbose:
            print("[slime-router] Shutdown complete")

    async def _check_one_worker(self, worker_url: str) -> Tuple[str, str]:
        """Performs a health check on a single worker."""
        try:
            response = await self.health_client.get(f"{worker_url}/health")
            if response.status_code == 200:
                return worker_url, "healthy"
        except httpx.RequestError as e:
            if self.verbose:
                print(f"[slime-router] Health check failed for {worker_url}: {type(e).__name__}")
        return worker_url, "unhealthy"

    async def background_health_check_task(self):
        """Periodically checks the health of all workers concurrently."""
        while True:
            try:
                async with self.worker_lock:
                    worker_urls = list(self.worker_urls.keys())

                if not worker_urls:
                    await asyncio.sleep(self.health_check_interval)
                    continue

                tasks = [self._check_one_worker(url) for url in worker_urls]
                results = await asyncio.gather(*tasks)

                async with self.worker_lock:
                    for worker_url, new_status in results:
                        old_status = self.worker_health.get(worker_url)
                        if old_status != new_status:
                            self.worker_health[worker_url] = new_status
                            if self.verbose:
                                print(f"[slime-router] Worker {worker_url} status changed to: {new_status}")

                jitter = random.uniform(-self.health_check_jitter, self.health_check_jitter)
                await asyncio.sleep(self.health_check_interval + jitter)

            except asyncio.CancelledError:
                if self.verbose:
                    print("[slime-router] Health check task cancelled, exiting...")
                break
            except Exception as e:
                if self.verbose:
                    print(f"[slime-router] Unexpected error in health check loop: {e}")
                await asyncio.sleep(5)

    def _update_weight_version_from_response(self, output):
        """
        Update weight version from SGLang response meta_info.
        This is the correct way to get weight version - from the generate response.
        """
        if "meta_info" not in output or "weight_version" not in output["meta_info"]:
            return

        current_weight_version = output["meta_info"]["weight_version"]

        # Update max_weight_version
        if self.max_weight_version is None or current_weight_version > self.max_weight_version:
            self.max_weight_version = current_weight_version
            if self.verbose:
                print(f"[slime-router] Updated max weight version to: {self.max_weight_version}")
        elif self.verbose:
            print(f"[slime-router] Current weight version {current_weight_version} <= max {self.max_weight_version}")

    def _setup_routes(self):
        """Setup all the HTTP routes"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.post("/remove_worker")(self.remove_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.post("/retrieve_from_text")(self.retrieve_from_text)
        self.app.get("/health")(self.health_check)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def health_check(self, request: Request):
        """Returns the health status of the router and its workers."""
        async with self.worker_lock:
            # Create Copy to avoid holding lock during json serialization
            worker_health_copy = self.worker_health.copy()
            worker_urls_copy = self.worker_urls.copy()
        return JSONResponse({"worker_health": worker_health_copy, "worker_connections": worker_urls_copy})

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        # Select worker under lock
        async with self.worker_lock:
            try:
                worker_url = self._use_url_unsafe()
            except RuntimeError as e:
                return JSONResponse(status_code=503, content={"error": str(e)})

        url = f"{worker_url}/{path}"

        # Get request body and headers
        body = await request.body()
        headers = dict(request.headers)

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=response.headers,
                media_type=response.headers.get("content-type"),
            )
        except httpx.RequestError as e:
            async with self.worker_lock:
                if self.worker_health.get(worker_url) != "unhealthy":
                    self.worker_health[worker_url] = "unhealthy"
                    if self.verbose:
                        print(f"[slime-router] Marked worker {worker_url} as unhealthy due to proxy error")
            return JSONResponse(
                status_code=503, content={"error": f"Worker {worker_url} is unavailable", "details": str(e)}
            )
        finally:
            async with self.worker_lock:
                self._finish_url_unsafe(worker_url)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker with body {"url": "http://127.0.0.1:10090"}
        """
        # 1) Prefer query param
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")

        # 2) Fallback to JSON body
        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        worker_url = worker_url.rstrip("/")

        # Initial health check (outside lock)
        _, initial_status = await self._check_one_worker(worker_url)

        async with self.worker_lock:
            if worker_url not in self.worker_urls:
                self.worker_urls[worker_url] = 0
                self.worker_health[worker_url] = initial_status
                if self.verbose:
                    print(f"[slime-router] Added new worker: {worker_url} (status: {initial_status})")
                return JSONResponse({"status": "success", "worker_url": worker_url, "initial_health": initial_status})
            else:
                return JSONResponse({"status": "already_exists", "worker_url": worker_url})

    async def remove_worker(self, request: Request):
        """Remove a worker from the router."""
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")

        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(status_code=400, content={"error": "worker_url is required"})

        worker_url = worker_url.rstrip("/")

        async with self.worker_lock:
            if worker_url in self.worker_urls:
                del self.worker_urls[worker_url]
                self.worker_health.pop(worker_url, None)
                if self.verbose:
                    print(f"[slime-router] Removed worker: {worker_url}")
                return JSONResponse({"status": "success", "worker_url": worker_url})
            else:
                return JSONResponse(status_code=404, content={"error": "Worker not found"})

    async def list_workers(self, request: Request):
        """List all registered workers"""
        async with self.worker_lock:
            return JSONResponse({"urls": list(self.worker_urls.keys())})

    async def retrieve_from_text(self, request: Request):
        """Get token information from text input"""
        body = await request.body()
        payload = json.loads(body) if body else {}

        text = payload.get("text", "")
        return_logp = payload.get("return_logp", False)

        # Use radix tree's retrieve_from_text method (no need to fetch weight version here)
        result = self.radix_tree.retrieve_from_text(text, return_logp=return_logp)

        # Handle the result based on whether logp was requested
        if return_logp:
            token_ids, logp = result
        else:
            token_ids = result
            logp = None

        result = {
            "tokens": token_ids,  # token IDs
            "response_length": len(token_ids),  # Length of response tokens
            "response": text,  # The input text
            "loss_mask": [],  # Loss mask for the tokens
        }

        # Add logp to response if requested
        if return_logp and logp is not None:
            result["logp"] = logp

        return result

    def _use_url_unsafe(self):
        """Select a healthy worker URL using least-connections. MUST be called with lock held."""
        healthy_workers = [
            url for url, status in self.worker_health.items() if status == "healthy" and url in self.worker_urls
        ]

        if not healthy_workers:
            raise RuntimeError("No healthy workers available")

        # get the url with minimal count from healthy workers
        url = min(healthy_workers, key=self.worker_urls.get)
        self.worker_urls[url] += 1
        return url

    def _finish_url_unsafe(self, url):
        """Mark the request to the given URL as finished. MUST be called with lock held."""
        if url in self.worker_urls:
            self.worker_urls[url] -= 1
            if self.worker_urls[url] < 0:
                if self.verbose:
                    print(f"[slime-router] WARNING: {url} count negative, resetting")
                self.worker_urls[url] = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use for tokenization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    # New arguments for health check configuration
    parser.add_argument("--health-check-interval", type=int, default=30, help="Health check interval in seconds")
    parser.add_argument("--health-check-timeout", type=float, default=5.0, help="Health check timeout in seconds")
    parser.add_argument("--health-check-jitter", type=float, default=2.0, help="Health check jitter in seconds")

    args = parser.parse_args()

    # Run the router
    run_router(args)
