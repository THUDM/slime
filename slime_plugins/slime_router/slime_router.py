import argparse
import json

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

# Import radix tree
from .radix_tree import StringRadixTrie


def run_slime_router(args):
    """
    Run the Slime router with the specified configuration.
    """
    # Initialize the router with tokenizer and lazy worker initialization
    slime_router = SlimeRouter(args, verbose=False)

    # Start the server
    uvicorn.run(slime_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class SlimeRouter:
    def __init__(self, args, verbose=False):
        """Initialize the SlimeRouter with SGLang router address"""
        self.args = args
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        self.app = FastAPI()

        # Worker information
        self.worker_urls: dict[str, int] = {}
        self.max_weight_version = None

        # TODO: remove this hardcode
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=16384),
            timeout=httpx.Timeout(None, connect=5.0),
        )

        # Initialize radix tree for caching with tokenizer (no router_url)
        self.radix_tree = StringRadixTrie(max_cache_size=10000, tokenizer=self.tokenizer, verbose=verbose)

        self._setup_routes()

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
                print(f"[SlimeRouter] Updated max weight version to: {self.max_weight_version}")
        elif self.verbose:
            print(f"[SlimeRouter] Current weight version {current_weight_version} <= max {self.max_weight_version}")

    def _setup_routes(self):
        """Setup all the HTTP routes"""
        # IMPORTANT: Register specific routes BEFORE the catch-all route
        self.app.post("/generate")(self.generate)
        self.app.post("/retrieve_from_text")(self.retrieve_from_text)
        self.app.get("/health")(self.health_check)
        self.app.post("/add_worker")(self.add_worker)
        self.app.get("/list_workers")(self.list_workers)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

        if self.verbose:
            print("set up complete")

    async def health_check(self, request: Request):
        # TODO: do health check
        return {}

    async def generate(self, request: Request):
        """Wrapper for SGLang router's /generate endpoint"""
        # Get the request body
        body = await request.body()
        payload = json.loads(body) if body else {}

        url = self._use_url()

        # Extract text from payload for radix tree operations
        input_text = payload.get("text", "")
        if self.verbose:
            print(f"[SlimeRouter] Received request with input_text: {input_text[:100]}...")

        # Ensure worker list is initialized
        if len(self.worker_urls) == 0:
            error_msg = "No workers available for processing requests"
            if self.verbose:
                print(f"[SlimeRouter] {error_msg}")
            return JSONResponse(status_code=503, content={"error": error_msg, "error_type": "no_workers_available"})

        # Get tokens for the input text from radix tree
        try:
            input_tokens, input_logprobs = self.radix_tree.retrieve_from_text(input_text, return_logp=True)
            if self.verbose:
                print(f"[SlimeRouter] Retrieved {len(input_tokens)} tokens from radix tree")
        except Exception as e:
            if self.verbose:
                print(f"[SlimeRouter] Error retrieving tokens from radix tree: {e}")
            return JSONResponse(status_code=500, content={"error": f"Failed to retrieve tokens: {str(e)}"})

        # Forward request to SGLang router
        try:
            # Modify the payload to use input_ids instead of text for token-in token-out
            sglang_payload = payload.copy()
            if input_text:
                # Replace "text" with "input_ids"
                sglang_payload.pop("text", None)
                sglang_payload["input_ids"] = input_tokens

            response = await self.client.post(f"{url}/generate", json=sglang_payload)
            response.raise_for_status()
            response_data = response.json()

            # Update weight version from SGLang response (correct way)
            self._update_weight_version_from_response(response_data)

            # Extract data for radix tree insertion
            if "text" in response_data and "output_ids" in response_data:
                generated_text = response_data["text"]
                generated_token_ids = response_data["output_ids"]

                # Combine input tokens and generated tokens
                full_text = input_text + generated_text

                # sglang will return the input token ids as well
                full_token_ids = generated_token_ids

                # Insert the full trajectory into radix tree with current weight version
                if full_text and full_token_ids:
                    try:
                        if "output_token_logprobs" in response_data.get("meta_info", {}):
                            generated_token_logprobs = [
                                item[0] for item in response_data["meta_info"]["output_token_logprobs"]
                            ]
                            full_logprobs = input_logprobs + generated_token_logprobs
                            self.radix_tree.insert(
                                full_text, full_token_ids, full_logprobs, weight_version=self.max_weight_version
                            )
                        else:
                            # Use default log probabilities (0.0) if not provided
                            self.radix_tree.insert(full_text, full_token_ids, weight_version=self.max_weight_version)

                        if self.verbose:
                            print(f"[SlimeRouter] Successfully cached trajectory with {len(full_token_ids)} tokens")
                    except Exception as e:
                        if self.verbose:
                            print(f"[SlimeRouter] Warning: Failed to cache trajectory: {e}")
                        # Don't fail the request if caching fails

            return response_data

        except Exception as e:
            error_msg = f"Error communicating with SGLang router: {str(e)}"
            if self.verbose:
                print(f"[SlimeRouter] {error_msg}")
            return JSONResponse(status_code=500, content={"error": error_msg, "error_type": "communication_error"})
        finally:
            self._finish_url(url)

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

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        # Forward all other paths to SGLang router
        worker_url = self._use_url()
        url = f"{worker_url}/{path}"

        # Get request body and headers
        body = await request.body()
        headers = dict(request.headers)

        if self.verbose:
            print(f"Proxying request to: {url}")
        timeout = httpx.Timeout(None)
        try:
            if request.method == "GET":
                response = await self.client.get(url, headers=headers)
            elif request.method == "POST":
                response = await self.client.post(url, content=body, headers=headers)
            elif request.method == "PUT":
                response = await self.client.put(url, content=body, headers=headers)
            elif request.method == "DELETE":
                response = await self.client.delete(url, headers=headers)
            else:
                return JSONResponse(status_code=405, content={"error": "Method not allowed"})

            # Try to return JSON response, fallback to text
            try:
                content = response.json()
            except:
                content = response.text

            return JSONResponse(status_code=response.status_code, content=content)
        except Exception as e:
            if self.verbose:
                print(f"Error in proxy endpoint: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
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
                print(f"[SlimeRouter] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_urls}

    async def list_workers(self, request: Request):
        """List all registered workers"""
        return {"urls": list(self.worker_urls.keys())}

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
    run_slime_router(args)
