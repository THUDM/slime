import json
from typing import Any, Dict, List

from fastapi import BaseHTTPMiddleware, FastAPI

from .openai_adapter import OpenAICompatibleToolCallAdapter


class OpenAIMiddleware(BaseHTTPMiddleware):
    """
    Simple middleware that converts sglang responses to OpenAI format.

    This middleware intercepts /generate responses and converts them using
    the OpenAICompatibleToolCallAdapter directly.
    """

    def __init__(self, app: FastAPI, *, router, tools_info: List[Dict[str, Any]] = None, parser_type: str = "qwen25"):
        super().__init__(app)
        self.router = router
        self.tools_info = tools_info or []
        self.parser_type = parser_type

        # Use adapter directly to convert response
        self.adapter = OpenAICompatibleToolCallAdapter(tools_info=self.tools_info, parser_type=self.parser_type)

        print(f"[OpenAI Middleware] Initialized with {len(self.tools_info)} " f"tools, parser: {self.parser_type}")

    async def dispatch(self, request, call_next):
        """Convert sglang responses to OpenAI format."""
        # Only process /generate endpoint
        if request.url.path != "/generate":
            return await call_next(request)

        # Get tools from request if provided
        try:
            request_json = await request.json()
            tools = request_json.get("tools", [])
            if tools:
                self.tools_info = tools
                print(f"[OpenAI Middleware] Updated tools: {len(tools)} tools")
        except Exception:
            pass  # Continue with existing tools if request parsing fails

        # Process request normally
        response = await call_next(request)

        # Convert response to OpenAI format
        if hasattr(response, "body") and response.body:
            try:
                response_data = json.loads(response.body.decode("utf-8"))

                # Get the generated text
                generated_text = response_data.get("text", "")

                # Parse with adapter
                parse_result = self.adapter.parse_response_to_openai_format(generated_text)

                if parse_result["success"]:
                    openai_message = parse_result["openai_message"]

                    # Create OpenAI format response
                    openai_response = {
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": openai_message.role,
                                    "content": openai_message.content,
                                    "tool_calls": [
                                        {"id": tc.id, "type": tc.type, "function": tc.function}
                                        for tc in (openai_message.tool_calls or [])
                                    ],
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "model": "sglang-model",
                        "id": f"chatcmpl-{hash(generated_text) % 1000000:06d}",
                        "object": "chat.completion",
                        "created": 1234567890,
                    }
                else:
                    # Fallback to simple text response
                    openai_response = {
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": generated_text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "model": "sglang-model",
                        "id": f"chatcmpl-{hash(generated_text) % 1000000:06d}",
                        "object": "chat.completion",
                        "created": 1234567890,
                    }

                # Update response body
                response.body = json.dumps(openai_response).encode("utf-8")
                print("[OpenAI Middleware] Successfully converted to " "OpenAI format")

            except Exception as e:
                print(f"[OpenAI Middleware] Error: {e}")
                # Return original response if conversion fails

        return response
