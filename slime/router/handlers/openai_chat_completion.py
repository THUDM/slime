"""
Simplified OpenAI Chat Completion API implementation for Slime Router.

This module provides 100% OpenAI-compatible Chat Completion API while leveraging
Slime Router's Radix Cache for optimal performance in multi-turn conversations.

Key Features:
- Full OpenAI API compatibility (text in/out)
- Unified flow: messages → generate → OpenAI format
- Radix Tree Middleware integration for automatic caching
- Simplified architecture with minimal abstraction

Architecture:
- Detect RadixTreeMiddleware presence
- Use query_cache_by_messages_template for semantic caching
- Forward to /generate endpoint for consistent processing
- Convert responses to OpenAI format
"""

import json
import time
import uuid
from typing import Optional, Tuple

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

# Try to import SGLang parsers for advanced output processing
try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    SGLANG_PARSERS_AVAILABLE = True
except ImportError:
    SGLANG_PARSERS_AVAILABLE = False
    ReasoningParser = None
    FunctionCallParser = None


class ChatCompletionHandler:
    """
    Chat Completion handler with auto cache detection.

    This handler automatically detects cache capability by testing the
    /retrieve_from_messages_template endpoint and uses the appropriate
    processing path:
    - With cache support: Use messages template caching
    - Without cache support: Direct proxy to SGLang
    """

    def __init__(self, router):
        """
        Initialize Chat Completion handler.

        Args:
            router: SlimeRouter instance for accessing middleware and workers
        """
        self.router = router
        self.args = router.args
        self._reasoning_parser = None  # Lazy-initialized reasoning parser
        self._function_call_parser = None  # Lazy-initialized function call parser

        # Cache frequently accessed components at initialization (performance optimization)
        # These are set once by middleware and never change, so safe to cache
        self._radix_tree = None
        self._tokenizer = None

    @property
    def radix_tree(self):
        """Lazy-load and cache radix tree (performance optimization)."""
        if self._radix_tree is None:
            self._radix_tree = self.router.component_registry.get("radix_tree")
        return self._radix_tree

    @property
    def tokenizer(self):
        """Lazy-load and cache tokenizer (performance optimization)."""
        if self._tokenizer is None:
            self._tokenizer = self.router.component_registry.get("tokenizer")
        return self._tokenizer

    async def handle_request(self, request: Request):
        """
        Handle Chat Completion request with auto cache detection.

        Args:
            request: FastAPI Request object

        Returns:
            JSON response (non-streaming only)
        """
        try:
            try:
                request_data = await request.json()
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON in request body: {str(e)}")

            # Validate request structure
            self._validate_chat_completion_request(request_data)

            # Check for streaming request and reject it
            if request_data.get("stream", False):
                raise HTTPException(
                    status_code=400,
                    detail="Streaming is not supported yet. Please set stream=false or remove the stream parameter.",
                )

            # Check if cache support is available (use router's method)
            cache_available = self.router._check_cache_availability()

            if not cache_available:
                # Direct mode: Proxy to SGLang Chat Completion API
                return await self._proxy_to_sglang_chat_from_data(request_data)

            # Cached mode: Direct flow with radix cache (no internal HTTP)
            return await self._handle_with_radix_cache(request_data)
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            # Catch any unexpected errors and log them
            import traceback

            error_traceback = traceback.format_exc()
            if getattr(self.args, "verbose", False):
                print(f"[slime-router] ERROR in handle_request: {e}")
                print(f"[slime-router] Traceback:\n{error_traceback}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def _validate_chat_completion_request(self, request_data: dict):
        """
        Minimal validation for Chat Completion request.

        Only validate absolutely required fields. Let SGLang handle
        detailed parameter validation and return appropriate errors.

        Args:
            request_data: Parsed request data

        Raises:
            HTTPException: If basic validation fails
        """
        # Only check the absolute minimum required for OpenAI API compatibility
        if "messages" not in request_data:
            raise HTTPException(status_code=400, detail="Invalid request: 'messages' field is required")

        messages = request_data["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            raise HTTPException(status_code=400, detail="Invalid request: 'messages' must be a non-empty list")

        # Basic message structure check - let SGLang handle detailed validation
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise HTTPException(
                    status_code=400, detail=f"Invalid request: message at index {i} must be a dictionary"
                )

            if "role" not in message:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: message at index {i} must have 'role' field",
                )

            # Normalize content to empty string if None/missing (following SGLang serving_chat.py:261-262)
            # This allows assistant messages with tool_calls to have null content per OpenAI spec
            if "content" not in message or message["content"] is None:
                message["content"] = ""

    def _parse_generated_output(
        self, generated_text: str, request_data: dict
    ) -> Tuple[str, Optional[dict], Optional[str]]:
        """
        Parse generated output with SGLang parsers (reasoning + tool calls).

        This method integrates SGLang's reasoning parser and function call parser
        to process the raw model output into structured format.

        Args:
            generated_text: Raw text from model generation
            request_data: Original request data (for tools, reasoning config)

        Returns:
            Tuple of (final_text, tool_calls, reasoning_text)
            - final_text: Text content to show to user (after parsing)
            - tool_calls: Parsed tool calls (if any)
            - reasoning_text: Extracted reasoning content (if any)
        """
        if not SGLANG_PARSERS_AVAILABLE:
            # Parsers not available - return raw text
            return generated_text, None, None

        final_text = generated_text
        tool_calls = None
        reasoning_text = None

        try:
            # Step 1: Parse reasoning content (if reasoning parser configured)
            reasoning_parser_type = getattr(self.args, "sglang_reasoning_parser", None)
            if reasoning_parser_type:
                if not self._reasoning_parser:
                    # Lazy initialize reasoning parser
                    self._reasoning_parser = ReasoningParser(
                        model_type=reasoning_parser_type,
                        stream_reasoning=False,  # For non-streaming, accumulate reasoning
                    )

                # Parse reasoning
                reasoning_text, normal_text = self._reasoning_parser.parse_non_stream(final_text)
                final_text = normal_text if normal_text else final_text

            # Step 2: Parse tool calls (if tools provided)
            tools = request_data.get("tools")
            tool_call_parser_type = getattr(self.args, "sglang_tool_call_parser", None)

            if tools and tool_call_parser_type:
                if not self._function_call_parser:
                    # Lazy initialize function call parser
                    from sglang.srt.entrypoints.openai.protocol import Tool

                    # Convert OpenAI tool format to SGLang Tool format if needed
                    sglang_tools = [Tool(**tool) if isinstance(tool, dict) else tool for tool in tools]
                    self._function_call_parser = FunctionCallParser(
                        tools=sglang_tools, tool_call_parser=tool_call_parser_type
                    )

                # Parse tool calls
                remaining_text, parsed_calls = self._function_call_parser.parse_non_stream(final_text)
                if parsed_calls:
                    final_text = remaining_text
                    # Convert SGLang ToolCallItem to OpenAI tool_calls format
                    tool_calls = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": (
                                    json.dumps(call.arguments) if isinstance(call.arguments, dict) else call.arguments
                                ),
                            },
                        }
                        for call in parsed_calls
                    ]

        except Exception as e:
            # Parser error - log and return raw text
            if getattr(self.args, "verbose", False):
                print(f"[slime-router] Warning: Parser error, using raw output: {e}")
            return generated_text, None, None

        return final_text, tool_calls, reasoning_text

    async def _proxy_to_sglang_chat_from_data(self, request_data: dict):
        """
        Direct proxy mode: Forward request data to SGLang Chat Completion API.

        This is a helper method for when we need to proxy from parsed data instead of a Request object.

        Args:
            request_data: Parsed request data

        Returns:
            Direct response from SGLang
        """
        worker_url = await self.router._use_url()
        sglang_url = f"{worker_url}/v1/chat/completions"

        try:
            # Non-streaming proxy with error mapping
            try:
                response = await self.router.client.request("POST", sglang_url, json=request_data)
                content = await response.aread()

                # Check for SGLang errors and map to OpenAI format
                if response.status_code >= 400:
                    await self._handle_sglang_error(response, content)

                return Response(content=content, status_code=response.status_code, headers=dict(response.headers))
            except httpx.HTTPStatusError as e:
                # Handle HTTP errors from SGLang
                await self._handle_sglang_error(e.response, await e.response.aread())
                raise
            except httpx.RequestError as e:
                # Handle connection/network errors
                raise HTTPException(
                    status_code=503, detail="Service temporarily unavailable: Unable to reach inference backend"
                )
        finally:
            await self.router._finish_url(worker_url)

    async def _handle_sglang_error(self, response, content):
        """
        Map SGLang errors to OpenAI-compatible error format.

        Args:
            response: HTTP response from SGLang
            content: Response content bytes
        """
        try:
            error_data = json.loads(content.decode("utf-8")) if content else {}

            # Map common SGLang errors to OpenAI format
            if response.status_code == 400:
                # Validation errors - pass through SGLang's message
                detail = error_data.get("error", error_data.get("detail", "Invalid request parameters"))
                raise HTTPException(status_code=400, detail=detail)
            elif response.status_code == 429:
                # Rate limiting
                raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
            elif response.status_code >= 500:
                # Server errors
                raise HTTPException(
                    status_code=response.status_code, detail="Inference service error. Please try again later."
                )
            else:
                # Other errors
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_data.get("error", error_data.get("detail", "Unknown error")),
                )
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If we can't parse the error, return a generic message
            raise HTTPException(status_code=response.status_code, detail="Service error: Unable to process request")

    async def _handle_with_radix_cache(self, request_data: dict):
        """
        Cached mode: Direct flow with radix cache (no internal HTTP).

        This method implements the Two-Path Architecture for cache-enabled mode:
        1. Apply chat template to get text
        2. Query radix cache to get token_ids
        3. Call SGLang /generate directly with tokens (token in, token out)
        4. Maintain radix cache with output tokens
        5. Parse output with tool call/reasoning parsers
        6. Convert to OpenAI chat.completion format

        Args:
            request_data: Parsed request data

        Returns:
            OpenAI-formatted response (non-streaming)
        """
        messages = request_data.get("messages", [])
        tools = request_data.get("tools", None)

        # Step 1: Get tokenizer and radix tree (cached, no lock contention after first call)
        try:
            radix_tree = self.radix_tree
            tokenizer = self.tokenizer

            # Step 2: Apply chat template to convert messages to text
            text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)

            if not text or not text.strip():
                raise RuntimeError("Messages template resulted in empty text")

            # Step 3: Query radix cache to get token_ids
            token_ids, _, _, _ = await radix_tree.get_or_create_tokenization_async(text)

            if not token_ids:
                raise RuntimeError("Failed to get tokens from radix tree")

        except Exception as e:
            if getattr(self.args, "verbose", False):
                print(f"[slime-router] Warning: Failed to get cached tokens, falling back to direct mode: {e}")
            # Fallback to direct proxy
            return await self._proxy_to_sglang_chat_from_data(request_data)

        # Step 4: Call SGLang /generate directly with tokens (token in, token out)
        sampling_params = self._build_sampling_params(request_data)

        return await self._non_stream_generate_with_cache(token_ids, sampling_params, radix_tree, text, request_data)

    def _build_sampling_params(
        self, request_data: dict
    ) -> dict:  # TODO this function should match with oai endpoint parameters
        """
        Build sampling parameters for SGLang generation request.

        Args:
            request_data: Parsed request data from Chat Completion API

        Returns:
            Dictionary of sampling parameters compatible with SGLang
        """
        sampling_params = {
            # Core generation parameters
            "max_new_tokens": request_data.get("max_tokens", 1024),
            "temperature": request_data.get("temperature", 1.0),
            "top_p": request_data.get("top_p", 1.0),
            "top_k": request_data.get("top_k", -1),
            "min_p": request_data.get("min_p", 0.0),
            # Penalty parameters
            "frequency_penalty": request_data.get("frequency_penalty", 0.0),
            "presence_penalty": request_data.get("presence_penalty", 0.0),
            # Stop conditions
            "stop": request_data.get("stop"),
            "stop_token_ids": request_data.get("stop_token_ids"),
            "ignore_eos": request_data.get("ignore_eos"),
            # Special token handling
            "skip_special_tokens": request_data.get("skip_special_tokens"),
            "spaces_between_special_tokens": request_data.get("spaces_between_special_tokens"),
            "no_stop_trim": request_data.get("no_stop_trim"),
        }

        # Remove None values to keep request clean and avoid SGLang errors
        # Note: 'stream' parameter is NOT part of sampling_params
        # It's handled separately in the request JSON for /generate endpoint
        return {k: v for k, v in sampling_params.items() if v is not None}

    async def _non_stream_generate_with_cache(
        self, token_ids: list, sampling_params: dict, radix_tree, input_text: str, request_data: dict
    ):
        """
        Non-streaming generation with direct SGLang call and cache maintenance.

        This method implements the cache-enabled path without internal HTTP:
        1. Call SGLang /generate directly with input_ids (token in)
        2. Get output_ids from response (token out)
        3. Maintain radix cache with output tokens
        4. Convert to OpenAI chat.completion format

        Args:
            token_ids: Input token IDs from radix cache
            sampling_params: Sampling parameters for generation
            radix_tree: RadixTree instance for cache maintenance
            input_text: Original input text (for cache maintenance)
            request_data: Original request data (for model name, etc.)

        Returns:
            JSONResponse: OpenAI-formatted chat.completion response
        """
        # Get a worker URL
        worker_url = await self.router._use_url()

        try:
            # Use router's shared client for consistency
            # Get timeout from args, default to 120s
            timeout = getattr(self.args, "slime_router_generation_timeout", 120.0)
            response = await self.router.client.post(
                f"{worker_url}/generate",
                json={
                    "input_ids": token_ids,
                    "sampling_params": sampling_params,
                    "return_logprob": True,  # Request logprobs
                    "return_text_in_logprobs": False,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            generate_data = response.json()

        except httpx.TimeoutException:
            timeout_val = getattr(self.args, "slime_router_generation_timeout", 120.0)
            raise HTTPException(
                status_code=504,
                detail=f"Request timeout after {timeout_val}s while calling SGLang worker. You can adjust this with --slime-router-generation-timeout parameter.",
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"SGLang worker error: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Failed to connect to SGLang worker: {str(e)}")
        finally:
            await self.router._finish_url(worker_url)

        # Extract generated text and token information from output_token_logprobs
        # Format: output_token_logprobs is a list of [logprob, token_id] pairs
        try:
            output_token_logprobs = generate_data.get("meta_info", {}).get("output_token_logprobs", [])
            output_ids = [item[1] for item in output_token_logprobs]
            output_logprobs = [float(item[0]) for item in output_token_logprobs]

            # Decode tokens to get response text
            if output_ids:
                generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
            else:
                # Fallback to text field if no logprobs available
                generated_text = generate_data.get("text", "")
        except (KeyError, IndexError, TypeError) as e:
            if getattr(self.args, "verbose", False):
                print(f"[slime-router] Warning: Failed to extract from output_token_logprobs: {e}, using fallback")
            # Fallback to original method
            generated_text = generate_data.get("text", "")
            output_ids = generate_data.get(
                "output_ids", self.tokenizer.encode(generated_text, add_special_tokens=False)
            )
            output_logprobs = len(output_ids) * [0.0]  # Dummy logprobs if not available

        # Maintain radix cache: insert full sequence (input + output)
        if output_ids:
            try:
                # Combine input and output tokens for cache insertion
                full_text = input_text + generated_text
                await radix_tree.insert(
                    full_text, token_ids + output_ids, output_logprobs
                )  # TODO implement insert async
            except Exception as e:
                if getattr(self.args, "verbose", False):
                    print(f"[slime-router] Warning: Failed to update radix cache: {e}")

        # Calculate token usage
        prompt_tokens = len(token_ids)
        completion_tokens = len(output_ids) if output_ids else len(generated_text.split())
        total_tokens = prompt_tokens + completion_tokens

        # Parse output with SGLang parsers (reasoning + tool calls)
        final_text, tool_calls, reasoning_text = self._parse_generated_output(generated_text, request_data)

        # Build message content
        message_content = {"role": "assistant", "content": final_text}

        # Add tool_calls if present
        if tool_calls:
            message_content["tool_calls"] = tool_calls

        # Optionally include reasoning in metadata (non-standard, for debugging)
        if reasoning_text and getattr(self.args, "include_reasoning_in_response", False):
            message_content["reasoning"] = reasoning_text

        if tool_calls:
            finish_reason = "tool_calls"
        else:
            try:
                finish_reason = generate_data.get("meta_info", {}).get("finish_reason", "stop")
            except Exception as e:
                if getattr(self.args, "verbose", False):
                    print(f"[slime-router] Warning: Failed to get finish_reason: {e}, defaulting to 'stop'")
                finish_reason = "stop"
        # Convert to OpenAI format
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": getattr(self.args, "model_name", "slime-model"),
            "choices": [
                {
                    "index": 0,
                    "message": message_content,
                    "finish_reason": finish_reason,  # Use extracted
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

        return JSONResponse(content=openai_response)


# Factory function for creating ChatCompletion handlers
def create_chat_completion_handler(router) -> ChatCompletionHandler:
    """
    Factory function to create Chat Completion handler.

    Args:
        router: SlimeRouter instance

    Returns:
        Configured Chat Completion handler
    """
    return ChatCompletionHandler(router)
