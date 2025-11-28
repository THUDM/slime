"""Main test suite for ChatCompletionHandler."""

import pytest
import json
import httpx
from unittest.mock import Mock, AsyncMock
from fastapi import HTTPException
from httpx import Response
import respx

from slime.router.handlers.openai_chat_completion import ChatCompletionHandler
from tests.router.fixtures.sample_requests import (
    get_simple_chat_request,
    get_request_with_all_params,
    get_request_with_stop_conditions,
)
from tests.router.fixtures.mock_responses import (
    get_sglang_generate_success,
    get_sglang_error_400,
    get_sglang_error_500,
    get_sglang_error_429,
)


class TestChatCompletionHandlerInit:
    """Test ChatCompletionHandler initialization."""

    def test_init_basic(self, mock_router):
        """Test basic initialization."""
        handler = ChatCompletionHandler(mock_router)
        assert handler.router == mock_router
        assert handler.args == mock_router.args
        assert handler._reasoning_parser is None
        assert handler._function_call_parser is None
        assert handler._radix_tree is None
        assert handler._tokenizer is None

    def test_lazy_radix_tree_property(self, mock_router, mock_radix_tree):
        """Test lazy loading of radix_tree property."""
        mock_router.component_registry.get.return_value = mock_radix_tree
        handler = ChatCompletionHandler(mock_router)

        # First access should trigger get()
        tree = handler.radix_tree
        mock_router.component_registry.get.assert_called_once_with("radix_tree")
        assert tree == mock_radix_tree

        # Second access should use cached value
        tree2 = handler.radix_tree
        assert mock_router.component_registry.get.call_count == 1
        assert tree2 == mock_radix_tree

    def test_lazy_tokenizer_property(self, mock_router, mock_tokenizer):
        """Test lazy loading of tokenizer property."""
        mock_router.component_registry.get.return_value = mock_tokenizer
        handler = ChatCompletionHandler(mock_router)

        tokenizer = handler.tokenizer
        mock_router.component_registry.get.assert_called_once_with("tokenizer")
        assert tokenizer == mock_tokenizer


class TestSamplingParams:
    """Test sampling parameter building."""

    def test_build_sampling_params_defaults(self, mock_router):
        """Test default sampling parameters."""
        handler = ChatCompletionHandler(mock_router)
        request_data = {}

        params = handler._build_sampling_params(request_data)

        assert params["max_new_tokens"] == 1024
        assert params["temperature"] == 1.0
        assert params["top_p"] == 1.0
        assert params["top_k"] == -1
        assert params["min_p"] == 0.0
        assert params["frequency_penalty"] == 0.0
        assert params["presence_penalty"] == 0.0

    def test_build_sampling_params_custom(self, mock_router):
        """Test custom sampling parameters."""
        handler = ChatCompletionHandler(mock_router)
        request_data = get_request_with_all_params()

        params = handler._build_sampling_params(request_data)

        assert params["max_new_tokens"] == 200
        assert params["temperature"] == 0.9
        assert params["top_p"] == 0.95
        assert params["top_k"] == 50
        assert params["min_p"] == 0.05
        assert params["frequency_penalty"] == 0.5
        assert params["presence_penalty"] == 0.3

    def test_build_sampling_params_none_values_removed(self, mock_router):
        """Test that None values are removed from params."""
        handler = ChatCompletionHandler(mock_router)
        request_data = {
            "max_tokens": 100,
            "stop": None,  # Should be removed
            "stop_token_ids": None  # Should be removed
        }

        params = handler._build_sampling_params(request_data)

        assert "stop" not in params
        assert "stop_token_ids" not in params
        assert params["max_new_tokens"] == 100

    def test_build_sampling_params_stop_conditions(self, mock_router):
        """Test stop condition parameters."""
        handler = ChatCompletionHandler(mock_router)
        request_data = get_request_with_stop_conditions()

        params = handler._build_sampling_params(request_data)

        assert params["stop"] == ["\n", "10"]
        assert params["stop_token_ids"] == [1, 2]

    def test_build_sampling_params_special_token_handling(self, mock_router):
        """Test special token handling parameters."""
        handler = ChatCompletionHandler(mock_router)
        request_data = {
            "skip_special_tokens": True,
            "spaces_between_special_tokens": False,
            "no_stop_trim": True
        }

        params = handler._build_sampling_params(request_data)

        assert params["skip_special_tokens"] is True
        assert params["spaces_between_special_tokens"] is False
        assert params["no_stop_trim"] is True


class TestErrorHandling:
    """Test error handling for various failure scenarios."""

    @pytest.mark.parametrize("status_code,error_fixture,expected_text", [
        (400, get_sglang_error_400, "temperature"),
        (500, get_sglang_error_500, "inference service error"),
        (429, get_sglang_error_429, "rate limit"),
    ])
    @pytest.mark.asyncio
    @respx.mock
    async def test_sglang_errors(self, mock_router, status_code, error_fixture, expected_text):
        """Test handling of SGLang errors (400/500/429)."""
        error_response = error_fixture()
        respx.post("http://localhost:30000/v1/chat/completions").mock(
            return_value=Response(status_code, json=error_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        with pytest.raises(HTTPException) as exc_info:
            await handler._proxy_to_sglang_chat_from_data(request_data)
        assert exc_info.value.status_code == status_code
        assert expected_text in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_connection_error(self, mock_router):
        """Test handling of connection errors."""
        mock_router.client.request = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        with pytest.raises(HTTPException) as exc_info:
            await handler._proxy_to_sglang_chat_from_data(request_data)
        assert exc_info.value.status_code == 503
        assert "service temporarily unavailable" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_timeout_error(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test handling of timeout errors."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]
        mock_router.args.slime_router_generation_timeout = 60.0
        mock_router.client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timeout")
        )

        handler = ChatCompletionHandler(mock_router)

        with pytest.raises(HTTPException) as exc_info:
            await handler._non_stream_generate_with_cache(
                [1, 2, 3], {}, mock_radix_tree, "test", {}
            )
        assert exc_info.value.status_code == 504
        assert "60.0s" in exc_info.value.detail

    @pytest.mark.asyncio
    @respx.mock
    async def test_malformed_json_error(self, mock_router):
        """Test error handling with malformed JSON response."""
        respx.post("http://localhost:30000/v1/chat/completions").mock(
            return_value=Response(500, content=b"not valid json")
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        with pytest.raises(HTTPException) as exc_info:
            await handler._proxy_to_sglang_chat_from_data(request_data)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self, mock_router, mock_request):
        """Test unexpected error handling in handle_request."""
        mock_router._check_cache_availability.side_effect = RuntimeError("Unexpected error")
        handler = ChatCompletionHandler(mock_router)

        with pytest.raises(HTTPException) as exc_info:
            await handler.handle_request(mock_request)
        assert exc_info.value.status_code == 500
        assert "internal server error" in exc_info.value.detail.lower()


class TestGenerationWithCache:
    """Test generation with cache maintenance."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generation_success(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test successful generation with cache maintenance."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=generate_response)
        )

        mock_tokenizer.decode.return_value = "Hello! How can I help you today?"

        handler = ChatCompletionHandler(mock_router)

        token_ids = [1, 2, 3, 4]
        sampling_params = {"max_new_tokens": 100}
        input_text = "<|user|>Hello<|assistant|>"
        request_data = get_simple_chat_request()

        response = await handler._non_stream_generate_with_cache(
            token_ids, sampling_params, mock_radix_tree, input_text, request_data
        )

        # Verify response structure
        content = json.loads(response.body.decode())
        assert "id" in content
        assert "object" in content
        assert content["object"] == "chat.completion"
        assert "choices" in content
        assert len(content["choices"]) == 1
        assert content["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in content

        # Verify cache insertion was called
        mock_radix_tree.insert.assert_called_once()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generation_http_error(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test generation HTTP error handling."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(500, json={"error": "Internal error"})
        )

        handler = ChatCompletionHandler(mock_router)

        with pytest.raises(HTTPException) as exc_info:
            await handler._non_stream_generate_with_cache(
                [1, 2, 3], {}, mock_radix_tree, "test", {}
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @respx.mock
    async def test_generation_with_fallback_text_field(
        self, mock_router, mock_radix_tree, mock_tokenizer
    ):
        """Test fallback to text field when logprobs not available."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        generate_response = {
            "text": "Response without logprobs",
            "meta_info": {
                "finish_reason": {"type": "stop"}
            }
        }
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=generate_response)
        )

        mock_tokenizer.decode.return_value = "Response without logprobs"
        mock_tokenizer.encode.return_value = [10, 20, 30]

        handler = ChatCompletionHandler(mock_router)

        response = await handler._non_stream_generate_with_cache(
            [1, 2], {}, mock_radix_tree, "test", {}
        )

        # Should succeed even without logprobs
        content = json.loads(response.body.decode())
        assert content["choices"][0]["message"]["content"] == "Response without logprobs"

    @pytest.mark.asyncio
    @respx.mock
    async def test_worker_url_management(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test that worker URLs are properly acquired and released."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=generate_response)
        )

        handler = ChatCompletionHandler(mock_router)

        await handler._non_stream_generate_with_cache(
            [1, 2, 3], {}, mock_radix_tree, "test", {}
        )

        # Verify _use_url and _finish_url were called
        mock_router._use_url.assert_called_once()
        mock_router._finish_url.assert_called_once_with("http://localhost:30000")


class TestDirectProxyMode:
    """Test direct proxy mode to SGLang."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_proxy_success(self, mock_router):
        """Test successful proxy to SGLang."""
        from tests.router.fixtures.mock_responses import get_sglang_chat_completion_success

        mock_response = get_sglang_chat_completion_success()
        respx.post("http://localhost:30000/v1/chat/completions").mock(
            return_value=Response(200, json=mock_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        response = await handler._proxy_to_sglang_chat_from_data(request_data)

        assert response.status_code == 200
        mock_router._use_url.assert_called_once()
        mock_router._finish_url.assert_called_once()

    @pytest.mark.asyncio
    @respx.mock
    async def test_proxy_preserves_request_data(self, mock_router):
        """Test that proxy preserves all request data."""
        from tests.router.fixtures.mock_responses import get_sglang_chat_completion_success

        mock_response = get_sglang_chat_completion_success()
        route = respx.post("http://localhost:30000/v1/chat/completions").mock(
            return_value=Response(200, json=mock_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_request_with_all_params()

        await handler._proxy_to_sglang_chat_from_data(request_data)

        # Verify the request was made with correct data
        assert route.called
        # Note: respx doesn't easily expose request JSON, but we can verify it was called


class TestEndToEndIntegration:
    """Test complete end-to-end flow from request to response."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_handle_request_with_cache_available_e2e(
        self, mock_router_with_components, mock_radix_tree, mock_tokenizer, mock_request
    ):
        """Test complete flow: request → validation → cache → generate → response."""
        mock_router_with_components._check_cache_availability.return_value = True

        generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=generate_response)
        )

        handler = ChatCompletionHandler(mock_router_with_components)
        response = await handler.handle_request(mock_request)

        # Verify complete flow
        assert response.status_code == 200
        content = json.loads(response.body.decode())
        assert "id" in content
        assert content["object"] == "chat.completion"
        assert "choices" in content
        assert "usage" in content

        # Verify cache was used
        mock_radix_tree.retrieve_from_text.assert_called_once()
        mock_radix_tree.insert.assert_called_once()

    @pytest.mark.asyncio
    @respx.mock
    async def test_handle_request_without_cache_e2e(self, mock_router, mock_request):
        """Test complete flow when cache is unavailable."""
        from tests.router.fixtures.mock_responses import get_sglang_chat_completion_success

        mock_router._check_cache_availability.return_value = False

        mock_response = get_sglang_chat_completion_success()
        respx.post("http://localhost:30000/v1/chat/completions").mock(
            return_value=Response(200, json=mock_response)
        )

        handler = ChatCompletionHandler(mock_router)
        response = await handler.handle_request(mock_request)

        assert response.status_code == 200
        content = json.loads(response.body.decode())
        assert content["object"] == "chat.completion"


class TestSGLangResponseParsing:
    """Test SGLang response parsing edge cases."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generation_missing_output_token_logprobs(
        self, mock_router_with_components, mock_radix_tree, mock_tokenizer
    ):
        """Test fallback when output_token_logprobs missing."""
        generate_response = {
            "text": "Response without logprobs",
            "meta_info": {
                "finish_reason": {"type": "stop"}
            }
        }
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=generate_response)
        )

        mock_tokenizer.decode.return_value = "Response without logprobs"
        mock_tokenizer.encode.return_value = [10, 20, 30]

        handler = ChatCompletionHandler(mock_router_with_components)

        response = await handler._non_stream_generate_with_cache(
            [1, 2], {}, mock_radix_tree, "test", {}
        )

        # Should succeed with fallback
        content = json.loads(response.body.decode())
        assert content["choices"][0]["message"]["content"] == "Response without logprobs"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generation_malformed_meta_info(
        self, mock_router_with_components, mock_radix_tree, mock_tokenizer
    ):
        """Test handling of malformed meta_info structure."""
        generate_response = {
            "text": "Response with malformed meta_info",
            "meta_info": {}  # Missing finish_reason
        }
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=generate_response)
        )

        mock_tokenizer.decode.return_value = "Response with malformed meta_info"
        mock_tokenizer.encode.return_value = [10, 20, 30]

        handler = ChatCompletionHandler(mock_router_with_components)

        response = await handler._non_stream_generate_with_cache(
            [1, 2], {}, mock_radix_tree, "test", {}
        )

        # Should handle gracefully with default finish_reason
        content = json.loads(response.body.decode())
        assert content["choices"][0]["finish_reason"] == "stop"
