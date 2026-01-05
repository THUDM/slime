import asyncio
import json

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from transformers import AutoTokenizer

from slime.utils.http_utils import post
from slime.utils.mask_utils import get_response_lengths
from slime.utils.types import Sample

from .radix_tree import StringRadixTrie

# Hop-by-hop headers that should not be forwarded
HOP_BY_HOP = {
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
}


def _filter_headers(headers):
    """Filter out hop-by-hop headers that should not be forwarded."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


async def _materialize_response(resp):
    """Convert streaming-like Response into a regular Response/JSONResponse safely."""
    # Collect all bytes from the streaming response
    body = b""
    async for chunk in resp.body_iterator:
        body += chunk

    # Try to parse as JSON based on content-type
    ct = resp.headers.get("content-type", "")
    headers = _filter_headers(resp.headers)

    if "application/json" in ct:
        # If it's JSON, try to parse and return as JSONResponse
        try:
            data = json.loads(body.decode("utf-8"))
            return JSONResponse(content=data, status_code=resp.status_code, headers=headers)
        except Exception:
            # JSON parsing failed, fall back to raw bytes
            pass

    # Other types: return as raw bytes (without content-length)
    return Response(content=body, status_code=resp.status_code, headers=headers, media_type=resp.media_type)


class RadixTreeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, router):
        super().__init__(app)
        self.router = router
        self.args = router.args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        self.radix_tree = StringRadixTrie(max_cache_size=10000, tokenizer=self.tokenizer, verbose=False)
        self.router.radix_tree = self.radix_tree

    async def dispatch(self, request: Request, call_next):

        path = request.url.path

        if path != "/generate":
            return await call_next(request)

        request_json = await request.json()
        if "text" in request_json:
            input_text = request_json.pop("text", "")
        elif "input_ids" in request_json:
            input_text = self.tokenizer.decode(request_json["input_ids"])
        else:
            input_text = None
        if not input_text:
            return await call_next(request)
        input_tokens, input_logprobs, input_loss_mask = self.radix_tree.retrieve_from_text(
            input_text, return_logprob=True
        )
        request_json["input_tokens"] = input_tokens
        request_json["stream"] = False
        request._json = request_json

        response_data = None
        for _ in range(5):
            response = await call_next(request)

            # If upstream returned a streaming response, materialize it to avoid Content-Length issues
            if response.__class__.__name__ == "_StreamingResponse":
                response = await _materialize_response(response)
            # Try to parse JSON from the current response for meta inspection
            try:
                if hasattr(response, "body") and isinstance(response.body, (bytes, bytearray)):
                    response_data = json.loads(response.body.decode("utf-8"))
                elif hasattr(response, "content") and isinstance(response.content, (dict, list)):
                    response_data = response.content  # JSONResponse.content is already a dict/list
            except Exception:
                response_data = None

            if (
                isinstance(response_data, dict)
                and "meta_info" in response_data
                and "finish_reason" in response_data["meta_info"]
                and response_data["meta_info"]["finish_reason"]["type"] != "abort"
            ):
                break
            # await 30 seconds for aborted responses
            await asyncio.sleep(30)

        if isinstance(response_data, dict) and "text" in response_data and "output_ids" in response_data:
            generated_text = response_data["text"]

            full_text = input_text + generated_text
            if full_text:
                try:
                    if "output_token_logprobs" in response_data.get("meta_info", {}):
                        generated_token_logprobs = [
                            item[0] for item in response_data["meta_info"]["output_token_logprobs"]
                        ]
                        generated_token_ids = [item[1] for item in response_data["meta_info"]["output_token_logprobs"]]
                        full_logprobs = input_logprobs + generated_token_logprobs
                        full_token_ids = input_tokens + generated_token_ids
                        full_loss_mask = input_loss_mask + [1] * len(generated_token_ids)
                        self.radix_tree.insert(
                            full_text,
                            full_token_ids,
                            full_logprobs,
                            full_loss_mask,
                            weight_version=response_data["meta_info"]["weight_version"],
                        )
                    else:
                        generated_token_ids = self.tokenizer(generated_text, add_special_tokens=False)["input_ids"]
                        full_token_ids = input_tokens + generated_token_ids
                        full_loss_mask = input_loss_mask + [1] * len(generated_token_ids)
                        self.radix_tree.insert(
                            full_text,
                            full_token_ids,
                            None,
                            full_loss_mask,
                            weight_version=response_data["meta_info"]["weight_version"],
                        )

                    if getattr(self.router, "verbose", False):
                        print(f"[slime-router] Successfully cached trajectory with {len(full_token_ids)} tokens")
                except Exception as e:
                    if getattr(self.router, "verbose", False):
                        print(f"[slime-router] Warning: Failed to cache trajectory: {e}")
        return response


async def postprocess_sample_with_radix_tree(args, sample: Sample, output: dict):
    assert not args.partial_rollout, "Currently partial rollout is not supported when using slime router"
    retrieve_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/retrieve_from_text"
    retrieve_payload = {"text": sample.prompt + output["text"], "return_logp": True}
    retrieve_output = await post(retrieve_url, retrieve_payload)
    sample.tokens = retrieve_output["tokens"]
    sample.response += output["text"]
    sample.loss_mask = retrieve_output["loss_mask"]
    sample.response_length = get_response_lengths([sample.loss_mask])[0]
    sample.loss_mask = sample.loss_mask[-sample.response_length :]
    sample.rollout_log_probs = retrieve_output["rollout_logp"][-sample.response_length :]
    return sample
