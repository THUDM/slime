import json
import time
import httpx
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import argparse

# Import radix tree
from .radix_tree import StringRadixTrie


def run_slime_router(args: argparse.Namespace):
    """
    Run the Slime router with the specified configuration.
    
    Args:
        args: Namespace object containing router configuration
    """
    # Initialize tokenizer if tokenizer name is provided
    tokenizer = None
    if args.tokenizer_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            if args.verbose:
                print(f"Loaded tokenizer: {args.tokenizer_name}")
        except ImportError:
            if args.verbose:
                print("Warning: transformers library not found. Tokenizer will not be available.")
        except Exception as e:
            if args.verbose:
                print(f"Warning: Failed to load tokenizer {args.tokenizer_name}: {e}")
    
    # Initialize the router with tokenizer and lazy worker initialization
    lazy_worker_init = getattr(args, 'lazy_worker_init', True)  # Default to True
    enable_weight_version = getattr(args, 'enable_weight_version', True)  # Default to True
    slime_router = SlimeRouter(
        args.sglang_host, 
        args.sglang_port, 
        tokenizer=tokenizer, 
        verbose=args.verbose,
        lazy_worker_init=lazy_worker_init,
        enable_weight_version=enable_weight_version
    )
    
    # Start the server
    uvicorn.run(
        slime_router.app, 
        host=args.host, 
        port=args.port,
        log_level="info"
    )


class SlimeRouter:
    def __init__(self, sglang_host: str, sglang_port: int, tokenizer=None, verbose=False, lazy_worker_init=True, enable_weight_version=True):
        """Initialize the SlimeRouter with SGLang router address"""
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.sglang_router_url = f"http://{sglang_host}:{sglang_port}"
        self.app = FastAPI()
        self.verbose = verbose
        self.lazy_worker_init = lazy_worker_init
        self.enable_weight_version = enable_weight_version
        
        # Worker information
        self.worker_urls = []
        self.max_weight_version = None
        self.worker_list_initialized = False
        
        # Initialize radix tree for caching with tokenizer (no router_url)
        self.radix_tree = StringRadixTrie(max_cache_size=10000, tokenizer=tokenizer, verbose=verbose)
        
        # Fetch worker list from router during initialization (unless lazy)
        if not lazy_worker_init:
            self._fetch_worker_list()
        
        self._setup_routes()
    
    def _fetch_worker_list(self, retry_on_empty=True, max_retries=10):
        """
        Fetch worker list from the SGLang router during initialization.
        This method is called during __init__ to populate worker information.
        
        Args:
            retry_on_empty: If True, retry when no workers are found (for startup scenarios)
            max_retries: Maximum number of retries when no workers are found
        """
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Fetch worker URLs
                list_workers_url = f"{self.sglang_router_url}/list_workers"
                if self.verbose:
                    print(f"[SlimeRouter] Fetching worker list from: {list_workers_url} (attempt {retry_count + 1})")
                    
                response = requests.get(list_workers_url, timeout=5)  # 增加超时时间
                response.raise_for_status()
                
                worker_data = response.json()
                self.worker_urls = worker_data.get("urls", [])
                
                if self.verbose:
                    print(f"[SlimeRouter] Successfully fetched {len(self.worker_urls)} workers: {self.worker_urls}")
                
                # retry when no workers are found
                if not self.worker_urls and retry_on_empty and retry_count < max_retries:
                    print(f"[SlimeRouter] No workers found, retrying in 2 seconds... (attempt {retry_count + 1}/{max_retries + 1})")
                    time.sleep(2)
                    retry_count += 1
                    continue
                elif not self.worker_urls:
                    print(f"[SlimeRouter] WARNING: No workers found in SGLang router at {self.sglang_router_url}")
                    print(f"[SlimeRouter] Please ensure SGLang workers are running and registered with the router")
                break 

            except requests.exceptions.RequestException as e:
                if self.verbose:
                    print(f"[SlimeRouter] Failed to fetch worker information from router: {e}")
                # Keep empty list if fetch fails
                self.worker_urls = []
                break
            except Exception as e:
                if self.verbose:
                    print(f"[SlimeRouter] Unexpected error while fetching worker information: {e}")
                self.worker_urls = []
                break
    

    def _ensure_worker_list_initialized(self):
        """
        Ensure worker list is initialized (for lazy initialization).
        """
        # If using lazy initialization and worker list not initialized, initialize it now
        if self.lazy_worker_init and not self.worker_list_initialized:
            if self.verbose:
                print("[SlimeRouter] Lazy initialization: fetching worker list for the first time")
            self._fetch_worker_list()
            self.worker_list_initialized = True
        
        # If worker list is empty, try to fetch new worker list
        if not self.worker_urls:
            if self.verbose:
                print("[SlimeRouter] Worker list is empty, attempting to refetch worker list")
            self._fetch_worker_list()

            # If still no workers after refetching, handle the error
            if not self.worker_urls:
                if self.verbose:
                    print("[SlimeRouter] No workers available after refetching worker list")
                return False
        
        return True
    
    def _update_weight_version_from_response(self, response_data):
        """
        Update weight version from SGLang response meta_info.
        This is the correct way to get weight version - from the generate response.
        """
        
        if not self.enable_weight_version:
            return
            
        # Use the first worker
        worker_url = self.worker_urls[0]
        
        try:
            get_weight_version_url = f"{worker_url}/get_weight_version"
            if self.verbose:
                print(f"[SlimeRouter] Fetching weight version from: {get_weight_version_url}")
                
            response = requests.get(get_weight_version_url, timeout=5)
            response.raise_for_status()
            
            version_data = response.json()
            current_weight_version = version_data.get("weight_version")
            
            if current_weight_version is not None:
                # Update max_weight_version
                if self.max_weight_version is None or current_weight_version > self.max_weight_version:
                    self.max_weight_version = current_weight_version
                    if self.verbose:
                        print(f"[SlimeRouter] Updated max weight version to: {self.max_weight_version}")
                elif self.verbose:
                    print(f"[SlimeRouter] Current weight version {current_weight_version} <= max {self.max_weight_version}")
            
        except requests.exceptions.RequestException as e:
            if self.verbose:
                print(f"[SlimeRouter] Failed to fetch weight version from worker: {e}")
        except Exception as e:
            if self.verbose:
                print(f"[SlimeRouter] Unexpected error while fetching weight version: {e}")
        
    def _setup_routes(self):
        """Setup all the HTTP routes"""
        # IMPORTANT: Register specific routes BEFORE the catch-all route
        self.app.post("/generate")(self.generate)
        self.app.post("/retrieve_from_text")(self.retrieve_from_text)
        self.app.get("/health")(self.health_check)
        self.app.get("/test")(self.test_endpoint)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

        if self.verbose:
            print("set up complete")
    
    async def health_check(self, request: Request):
        """Health check endpoint to verify SlimeRouter and SGLang router status"""
        health_status = {
            "slime_router": "healthy",
            "sglang_router_url": self.sglang_router_url,
            "worker_count": len(self.worker_urls),
            "workers": self.worker_urls,
            "max_weight_version": self.max_weight_version,
            "radix_tree_size": len(self.radix_tree.cache) if hasattr(self.radix_tree, 'cache') else "unknown"
        }
        
        # Test SGLang router connectivity
        try:
            import requests
            response = requests.get(f"{self.sglang_router_url}/list_workers", timeout=5)
            if response.status_code == 200:
                health_status["sglang_router"] = "healthy"
                sglang_data = response.json()
                health_status["sglang_workers"] = sglang_data.get("urls", [])
            else:
                health_status["sglang_router"] = f"unhealthy (status: {response.status_code})"
        except Exception as e:
            health_status["sglang_router"] = f"unreachable: {str(e)}"
        
        return health_status
    
    async def test_endpoint(self, request: Request):
        """Simple test endpoint to verify SlimeRouter is working"""
        return {
            "status": "ok",
            "message": "SlimeRouter is running",
            "sglang_router_url": self.sglang_router_url,
            "worker_count": len(self.worker_urls),
            "workers": self.worker_urls
        }
        
    async def generate(self, request: Request):
        """Wrapper for SGLang router's /generate endpoint"""
        # Get the request body
        body = await request.body()
        payload = json.loads(body) if body else {}
        
        # Extract text from payload for radix tree operations
        input_text = payload.get("text", "")
        if self.verbose:
            print(f'[SlimeRouter] Received request with input_text: {input_text[:100]}...')
        
        # Ensure worker list is initialized
        if not self._ensure_worker_list_initialized():
            error_msg = "No workers available for processing requests"
            if self.verbose:
                print(f"[SlimeRouter] {error_msg}")
            return JSONResponse(
                status_code=503,
                content={"error": error_msg, "error_type": "no_workers_available"}
            )

        # Get tokens for the input text from radix tree
        try:
            input_tokens, input_logprobs = self.radix_tree.retrieve_from_text(input_text, return_logp=True)
            if self.verbose:
                print(f"[SlimeRouter] Retrieved {len(input_tokens)} tokens from radix tree")
        except Exception as e:
            if self.verbose:
                print(f"[SlimeRouter] Error retrieving tokens from radix tree: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to retrieve tokens: {str(e)}"}
            )
        
        # Forward request to SGLang router
        timeout = httpx.Timeout(None)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                # Modify the payload to use input_ids instead of text for token-in token-out
                sglang_payload = payload.copy()
                if input_text:
                    # Replace "text" with "input_ids" 
                    sglang_payload.pop("text", None)
                    sglang_payload["input_ids"] = input_tokens
                
                if self.verbose:
                    print("=============== SGLang Payload: ========================")
                    print(f"input_ids length: {len(sglang_payload.get('input_ids', []))}")
                    print(f"sampling_params: {sglang_payload.get('sampling_params', {})}")

                response = await client.post(
                    f"{self.sglang_router_url}/generate",
                    json=sglang_payload
                )
                
                if response.status_code != 200:
                    error_msg = f"SGLang router returned status {response.status_code}: {response.text}"
                    if self.verbose:
                        print(f"[SlimeRouter] {error_msg}")
                    return JSONResponse(
                        status_code=response.status_code,
                        content={"error": error_msg, "error_type": "sglang_router_error"}
                    )
                
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON response from SGLang router: {str(e)}"
                    if self.verbose:
                        print(f"[SlimeRouter] {error_msg}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": error_msg}
                    )

                if self.verbose:
                    print("=============== SGLang Response: ========================")
                    print(f"Response keys: {list(response_data.keys())}")
                    if "text" in response_data:
                        print(f"Generated text length: {len(response_data['text'])}")
                
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
                                generated_token_logprobs = [item[0] for item in response_data["meta_info"]["output_token_logprobs"]]
                                full_logprobs = input_logprobs + generated_token_logprobs
                                self.radix_tree.insert(full_text, full_token_ids, full_logprobs, weight_version=self.max_weight_version)
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
                
            except httpx.TimeoutException as e:
                error_msg = f"Timeout connecting to SGLang router: {str(e)}"
                if self.verbose:
                    print(f"[SlimeRouter] {error_msg}")
                return JSONResponse(
                    status_code=504,
                    content={"error": error_msg, "error_type": "timeout"}
                )
            except httpx.ConnectError as e:
                error_msg = f"Cannot connect to SGLang router at {self.sglang_router_url}: {str(e)}"
                if self.verbose:
                    print(f"[SlimeRouter] {error_msg}")
                return JSONResponse(
                    status_code=503,
                    content={"error": error_msg, "error_type": "connection_error"}
                )
            except Exception as e:
                error_msg = f"Error communicating with SGLang router: {str(e)}"
                if self.verbose:
                    print(f"[SlimeRouter] {error_msg}")
                return JSONResponse(
                    status_code=500,
                    content={"error": error_msg, "error_type": "communication_error"}
                )
    
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
            "loss_mask": []  # Loss mask for the tokens
        }
        
        # Add logp to response if requested
        if return_logp and logp is not None:
            result["logp"] = logp
            
        return result
    
    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        # Forward all other paths to SGLang router
        url = f"{self.sglang_router_url}/{path}"
        
        # Get request body and headers
        body = await request.body()
        headers = dict(request.headers)
        
        if self.verbose:
            print(f"Proxying request to: {url}")
        timeout = httpx.Timeout(None)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                if request.method == "GET":
                    response = await client.get(url, headers=headers)
                elif request.method == "POST":
                    response = await client.post(url, content=body, headers=headers)
                elif request.method == "PUT":
                    response = await client.put(url, content=body, headers=headers)
                elif request.method == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    return JSONResponse(
                        status_code=405,
                        content={"error": "Method not allowed"}
                    )
                
                # Try to return JSON response, fallback to text
                try:
                    content = response.json()
                except:
                    content = response.text
                    
                return JSONResponse(
                    status_code=response.status_code,
                    content=content
                )
            except Exception as e:
                if self.verbose:
                    print(f"Error in proxy endpoint: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )


if __name__ == "__main__":
    import uvicorn
    import argparse
    
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