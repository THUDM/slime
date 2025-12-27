import argparse
import asyncio
import json
import threading

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from slime.router.utils.component_registry import ComponentRegistry
from slime.utils.misc import load_function


def run_router(args):
    """
    Run the Slime router with the specified configuration.
    """
    # Initialize the router with tokenizer and lazy worker initialization
    slime_router = SlimeRouter(args, verbose=args.verbose)

    # Start the server
    uvicorn.run(slime_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class SlimeRouter:
    def __init__(self, args, verbose=False):
        """Initialize the slime-router with SGLang router address"""
        self.args = args
        self.verbose = verbose

        self.app = FastAPI()

        # Initialize component registry for dependency injection
        self._component_registry = None
        self._registry_lock = threading.RLock()  # Use RLock for safer recursive access

        # Cache availability check result (None = not checked yet)
        self._cache_available = None
        self._cache_lock = threading.RLock()  # Use RLock for safer recursive access

        # Lazy-initialized chat completion handler (singleton)
        self._chat_completion_handler = None

        # Worker information
        self.worker_urls: dict[str, int] = {}
        self.max_weight_version = None

        max_connections = getattr(args, "slime_router_max_connections", None)
        if max_connections is None:
            max_connections = (
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )

        timeout = getattr(args, "slime_router_timeout", None)
        self._url_lock = asyncio.Lock()

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )

        self._setup_routes()

        for middleware_path in args.slime_router_middleware_paths or []:
            if self.verbose:
                print(f"[slime-router] Loading middleware from: {middleware_path}")
            middleware = load_function(middleware_path)
            self.app.add_middleware(middleware, router=self)

    def _setup_routes(self):
        """Setup all the HTTP routes"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.post("/retrieve_from_text")(self.retrieve_from_text)
        self.app.post("/retrieve_from_messages_template")(self.retrieve_from_messages_template)

        # OpenAI Chat Completion API (non-streaming only for now)
        self.app.post("/v1/chat/completions")(self.chat_completions)

        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    def get_component_registry(self) -> ComponentRegistry:
        """
        Get or create the component registry (thread-safe lazy initialization).

        Returns:
            ComponentRegistry instance
        """
        if self._component_registry is None:
            with self._registry_lock:
                # Double-check locking pattern
                if self._component_registry is None:
                    self._component_registry = ComponentRegistry()
        return self._component_registry

    @property
    def component_registry(self) -> ComponentRegistry:
        """Property accessor for component registry."""
        return self.get_component_registry()

    def _check_cache_availability(self) -> bool:
        """
        Check if cache support is available (thread-safe).

        This method checks if the router has radix tree caching enabled
        by looking for the radix_tree attribute or checking the component registry.

        Returns:
            bool: True if cache is available, False otherwise
        """
        if self._cache_available is not None:
            return self._cache_available

        with self._cache_lock:
            # Double-check locking
            if self._cache_available is not None:
                return self._cache_available

            # Check if radix tree is available in component registry
            has_radix_tree = self.component_registry.has("radix_tree")

            self._cache_available = has_radix_tree

            if self.verbose:
                print(f"[slime-router] Cache availability check: {has_radix_tree}")

            return self._cache_available

    async def health_check(self, request: Request):
        # TODO: do health check in background
        pass

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router (non-streaming)"""
        # Forward all other paths to SGLang router
        worker_url = await self._use_url()
        url = f"{worker_url}/{path}"

        # Get request body and headers
        body = await request.body()
        headers = dict(request.headers)

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
            # Eagerly read content so we can return JSON (not streaming)
            content = await response.aread()
            content_type = response.headers.get("content-type", "")
            try:
                # Prefer parsing JSON if possible
                data = json.loads(content)
                return JSONResponse(
                    content=data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            except Exception:
                # Fall back to raw body with original content type
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=content_type or None,
                )

        finally:
            await self._finish_url(worker_url)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
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

        # Add if new, keep a simple request count per worker
        if worker_url not in self.worker_urls:
            self.worker_urls[worker_url] = 0
            if self.verbose:
                print(f"[slime-router] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_urls}

    async def list_workers(self, request: Request):
        """List all registered workers"""
        return {"urls": list(self.worker_urls.keys())}

    async def retrieve_from_text(self, request: Request):
        """Get token information from text input"""
        body = await request.body()
        payload = json.loads(body) if body else {}

        text = payload.get("text", "")

        # Get radix tree from component registry
        radix_tree = self.component_registry.get("radix_tree")
        # Handle empty string early - return empty response
        if not text:
            return {
                "tokens": [],
                "response": text,
                "loss_mask": [],
                "token_length": 0,
                "loss_mask_length": 0,
                "rollout_logp": [],
            }

        # Use radix tree's retrieve_from_text method (no need to fetch weight version here)
        token_ids, logp, loss_mask = radix_tree.retrieve_from_text(text, return_logprob=True)

        # Handle the result based on whether logp was requested
        result = {
            "tokens": token_ids,  # token IDs
            "response": text,  # The input text
            "loss_mask": loss_mask,  # Loss mask for the tokens
            "token_length": len(token_ids),
            "loss_mask_length": len(loss_mask),
            "rollout_logp": logp,
        }

        return result

    def _use_url(self):
        """Select a worker URL using round-robin strategy"""
        assert len(self.worker_urls) > 0, "No workers available"

        async with self._url_lock:
            # get the url with minimal count
            url = min(self.worker_urls, key=self.worker_urls.get)
            self.worker_urls[url] += 1
            return url

    async def _finish_url(self, url):
        """Mark the request to the given URL as finished (async with lock)"""
        assert url in self.worker_urls, f"URL {url} not recognized"
        async with self._url_lock:
            self.worker_urls[url] -= 1
            assert self.worker_urls[url] >= 0, f"URL {url} count went negative"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use for tokenization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run the router
    run_router(args)
