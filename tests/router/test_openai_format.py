"""Tests for OpenAI API format compatibility."""

import pytest
import json
from unittest.mock import Mock, AsyncMock
from httpx import Response
import respx

from slime.router.handlers.openai_chat_completion import ChatCompletionHandler
from tests.router.fixtures.sample_requests import get_simple_chat_request
from tests.router.fixtures.mock_responses import get_sglang_generate_success


def assert_openai_response_format(response_data):
    """Validate OpenAI response format completeness."""
    # Top-level required fields
    assert "id" in response_data, "Response missing 'id' field"
    assert "object" in response_data, "Response missing 'object' field"
    assert response_data["object"] == "chat.completion", "Incorrect object type"
    assert "created" in response_data, "Response missing 'created' field"
    assert "model" in response_data, "Response missing 'model' field"
    assert "choices" in response_data, "Response missing 'choices' field"
    assert "usage" in response_data, "Response missing 'usage' field"

    # Validate choices structure
    assert isinstance(response_data["choices"], list), "choices must be a list"
    assert len(response_data["choices"]) > 0, "choices cannot be empty"

    choice = response_data["choices"][0]
    assert "index" in choice, "choice missing 'index' field"
    assert "message" in choice, "choice missing 'message' field"
    assert "finish_reason" in choice, "choice missing 'finish_reason' field"

    # Validate message structure
    message = choice["message"]
    assert "role" in message, "message missing 'role' field"
    assert message["role"] == "assistant", "message role must be 'assistant'"
    assert "content" in message, "message missing 'content' field"

    # Validate usage structure
    usage = response_data["usage"]
    assert "prompt_tokens" in usage, "usage missing 'prompt_tokens'"
    assert "completion_tokens" in usage, "usage missing 'completion_tokens'"
    assert "total_tokens" in usage, "usage missing 'total_tokens'"

    # Validate token counts
    assert isinstance(usage["prompt_tokens"], int), "prompt_tokens must be int"
    assert isinstance(usage["completion_tokens"], int), "completion_tokens must be int"
    assert isinstance(usage["total_tokens"], int), "total_tokens must be int"
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], \
        "total_tokens must equal sum of prompt_tokens and completion_tokens"


class TestOpenAIFormatCompatibility:
    """Test OpenAI API format compatibility."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_response_has_all_required_fields(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test that response contains all required OpenAI fields."""
        # Setup mocks
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        # Execute
        response = await handler._handle_with_radix_cache(request_data)

        # Parse response
        response_data = json.loads(response.body.decode())

        # Validate format
        assert_openai_response_format(response_data)

    @pytest.mark.asyncio
    @respx.mock
    async def test_finish_reason_stop(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test finish_reason is correctly set to 'stop'."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        response = await handler._handle_with_radix_cache(request_data)
        response_data = json.loads(response.body.decode())

        assert response_data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    @respx.mock
    async def test_finish_reason_tool_calls(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test finish_reason is 'tool_calls' when tool calls present."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)

        # Mock parser to return tool calls
        handler._parse_generated_output = Mock(return_value=(
            "",  # text
            [{"id": "call_123", "type": "function", "function": {"name": "get_weather"}}],  # tool_calls
            None  # reasoning
        ))

        request_data = get_simple_chat_request()
        response = await handler._handle_with_radix_cache(request_data)
        response_data = json.loads(response.body.decode())

        assert response_data["choices"][0]["finish_reason"] == "tool_calls"
        assert "tool_calls" in response_data["choices"][0]["message"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_tool_calls_format(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test that tool_calls follow OpenAI format."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)

        # Mock tool calls
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "SF"})
                }
            }
        ]
        handler._parse_generated_output = Mock(return_value=("", tool_calls, None))

        request_data = get_simple_chat_request()
        response = await handler._handle_with_radix_cache(request_data)
        response_data = json.loads(response.body.decode())

        message = response_data["choices"][0]["message"]
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1

        tool_call = message["tool_calls"][0]
        assert "id" in tool_call
        assert "type" in tool_call
        assert tool_call["type"] == "function"
        assert "function" in tool_call
        assert "name" in tool_call["function"]
        assert "arguments" in tool_call["function"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_usage_tokens_accurate(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test that token usage counts are accurate."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        # Setup with known token counts
        input_tokens = [1, 2, 3, 4]  # 4 tokens
        output_tokens = [10, 20, 30, 40, 50]  # 5 tokens

        mock_radix_tree.retrieve_from_text = Mock(
            return_value=(input_tokens, [0.0] * 4, [0] * 4)
        )

        mock_generate_response = {
            "text": "response",
            "meta_info": {
                "output_token_logprobs": [[-0.1, tid] for tid in output_tokens],
                "finish_reason": {"type": "stop"}
            }
        }
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        response = await handler._handle_with_radix_cache(request_data)
        response_data = json.loads(response.body.decode())

        usage = response_data["usage"]
        assert usage["prompt_tokens"] == 4
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 9

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_name_in_response(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test that model name is included in response."""
        mock_router.args.model_name = "custom-test-model"
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        response = await handler._handle_with_radix_cache(request_data)
        response_data = json.loads(response.body.decode())

        assert response_data["model"] == "custom-test-model"

    @pytest.mark.asyncio
    @respx.mock
    async def test_message_role_is_assistant(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test that message role is always 'assistant'."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        response = await handler._handle_with_radix_cache(request_data)
        response_data = json.loads(response.body.decode())

        message = response_data["choices"][0]["message"]
        assert message["role"] == "assistant"

    @pytest.mark.asyncio
    @respx.mock
    async def test_id_is_unique(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test that each response has a unique ID."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        # Generate two responses
        response1 = await handler._handle_with_radix_cache(request_data)
        response2 = await handler._handle_with_radix_cache(request_data)

        data1 = json.loads(response1.body.decode())
        data2 = json.loads(response2.body.decode())

        # IDs should be different
        assert data1["id"] != data2["id"]

        # Both should start with "chatcmpl-"
        assert data1["id"].startswith("chatcmpl-")
        assert data2["id"].startswith("chatcmpl-")
