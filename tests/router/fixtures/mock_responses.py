"""Mock responses for testing SGLang interactions."""


def get_sglang_generate_success():
    """Mock successful SGLang /generate response."""
    return {
        "text": "Hello! How can I help you today?",
        "output_ids": [10, 20, 30, 40, 50],
        "meta_info": {
            "output_token_logprobs": [
                [-0.5, 10],
                [-0.3, 20],
                [-0.4, 30],
                [-0.2, 40],
                [-0.6, 50]
            ],
            "finish_reason": {"type": "stop"},
            "weight_version": 1
        }
    }


def get_sglang_generate_with_tool_calls():
    """Mock SGLang /generate response with tool calls."""
    return {
        "text": "<tool_call>get_weather({\"location\": \"San Francisco\"})</tool_call>",
        "output_ids": [10, 20, 30],
        "meta_info": {
            "output_token_logprobs": [
                [-0.5, 10],
                [-0.3, 20],
                [-0.4, 30]
            ],
            "finish_reason": {"type": "stop"},
            "weight_version": 1
        }
    }


def get_sglang_chat_completion_success():
    """Mock successful SGLang chat completion response."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }


def get_sglang_error_400():
    """Mock SGLang 400 error response."""
    return {
        "error": "Invalid request: temperature must be between 0 and 2"
    }


def get_sglang_error_500():
    """Mock SGLang 500 error response."""
    return {
        "error": "Internal server error during model inference"
    }


def get_sglang_error_429():
    """Mock SGLang 429 rate limit error response."""
    return {
        "error": "Rate limit exceeded"
    }
