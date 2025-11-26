"""Sample OpenAI-format requests for testing."""


def get_simple_chat_request():
    """Get a simple chat completion request."""
    return {
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }


def get_multi_message_request():
    """Get a chat request with multiple messages."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather like?"},
            {"role": "assistant", "content": "I don't have access to weather data."},
            {"role": "user", "content": "Can you help me with something else?"}
        ],
        "max_tokens": 150,
        "temperature": 0.8
    }


def get_request_with_tools():
    """Get a chat request with tool definitions."""
    return {
        "messages": [
            {"role": "user", "content": "What is the weather in San Francisco?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "max_tokens": 100
    }


def get_request_with_stop_conditions():
    """Get a chat request with stop conditions."""
    return {
        "messages": [
            {"role": "user", "content": "Count to 10"}
        ],
        "max_tokens": 100,
        "stop": ["\n", "10"],
        "stop_token_ids": [1, 2]
    }


def get_request_with_all_params():
    """Get a chat request with all possible parameters."""
    return {
        "messages": [
            {"role": "user", "content": "Test message"}
        ],
        "max_tokens": 200,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 50,
        "min_p": 0.05,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "stop": ["STOP"],
        "stop_token_ids": [100],
    }


def get_invalid_request_no_messages():
    """Get an invalid request missing the messages field."""
    return {
        "max_tokens": 100
    }


def get_invalid_request_empty_messages():
    """Get an invalid request with empty messages list."""
    return {
        "messages": [],
        "max_tokens": 100
    }


def get_invalid_request_messages_not_list():
    """Get an invalid request where messages is not a list."""
    return {
        "messages": "not a list",
        "max_tokens": 100
    }


def get_request_with_null_content():
    """Get a request with null content (valid for tool calls)."""
    return {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}
                    }
                ]
            },
            {
                "role": "tool",
                "content": "Sunny, 72F",
                "tool_call_id": "call_123"
            }
        ]
    }


def get_streaming_request():
    """Get a streaming request (should be rejected)."""
    return {
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": True,
        "max_tokens": 100
    }
