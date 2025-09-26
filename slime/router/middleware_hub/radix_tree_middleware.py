from fastapi import BaseHTTPMiddleware, FastAPI
from transformers import AutoTokenizer

from .radix_tree import StringRadixTrie


class RadixTreeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, *, router):
        super().__init__(app)
        self.router = router
        self.args = router.args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.radix_tree = StringRadixTrie(max_cache_size=10000, tokenizer=self.tokenizer, verbose=False)

    async def dispatch(self, request, call_next):
        # Example middleware logic using radix tree
        path = request.url.path
        if path != "/generate":
            return await call_next(request)

        # pop "text" from request json and get input tokens from self.radix_tree and then use call_next
        request_json = await request.json()
        input_text = request_json.pop("text", "")
        if not input_text:
            return await call_next(request)
        input_tokens, input_logprobs = self.radix_tree.retrieve_from_text(input_text, return_logprob=True)
        request_json["input_tokens"] = input_tokens
        request._json = request_json  # Update the request json
        response = await call_next(request)

        # Extract data for radix tree insertion
        if "text" in response and "output_ids" in response:
            generated_text = response["text"]
            generated_token_ids = response["output_ids"]

            # Combine input tokens and generated tokens
            full_text = input_text + generated_text

            # sglang will return the input token ids as well
            full_token_ids = generated_token_ids

            # Insert the full trajectory into radix tree with current weight version
            if full_text and full_token_ids:
                try:
                    if "output_token_logprobs" in response.get("meta_info", {}):
                        generated_token_logprobs = [item[0] for item in response["meta_info"]["output_token_logprobs"]]
                        full_logprobs = input_logprobs + generated_token_logprobs
                        self.radix_tree.insert(
                            full_text, full_token_ids, full_logprobs, weight_version=self.max_weight_version
                        )
                    else:
                        # Use default log probabilities (0.0) if not provided
                        self.radix_tree.insert(full_text, full_token_ids, weight_version=self.max_weight_version)

                    if self.verbose:
                        print(f"[slime-router] Successfully cached trajectory with {len(full_token_ids)} tokens")
                except Exception as e:
                    if self.verbose:
                        print(f"[slime-router] Warning: Failed to cache trajectory: {e}")
                    # Don't fail the request if caching fails
        return response
