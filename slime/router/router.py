import argparse
import json

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

        # Worker information
        self.worker_urls: dict[str, int] = {}
        self.max_weight_version = None

        # TODO: remove this hardcode
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=args.sglang_server_concurrency
                * args.rollout_num_gpus
                // args.rollout_num_gpus_per_engine
            ),
            timeout=httpx.Timeout(None),
        )

        self._setup_routes()

        for middleware_path in args.slime_router_middleware_paths or []:
            if self.verbose:
                print(f"[slime-router] Loading middleware from: {middleware_path}")
            middleware = load_function(middleware_path)
            self.app.add_middleware(middleware, router=self)

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
        self.app.get("/list_workers")(self.list_workers)
        self.app.post("/retrieve_from_text")(self.retrieve_from_text)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def health_check(self, request: Request):
        # TODO: do health check in background
        pass

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        # Forward all other paths to SGLang router
        worker_url = self._use_url()
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

        finally:
            self._finish_url(worker_url)

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

    def _use_url(self):
        """Select a worker URL using round-robin strategy"""
        assert len(self.worker_urls) > 0, "No workers available"

        # get the url with mininal count
        url = min(self.worker_urls, key=self.worker_urls.get)
        self.worker_urls[url] += 1
        return url

    def _finish_url(self, url):
        """Mark the request to the given URL as finished"""
        assert url in self.worker_urls, f"URL {url} not recognized"
        self.worker_urls[url] -= 1
        assert self.worker_urls[url] >= 0, f"URL {url} count went negative"


if __name__ == "__main__":
    import argparse

    import uvicorn

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
