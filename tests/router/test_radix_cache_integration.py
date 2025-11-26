"""Tests for radix cache integration with chat completion handler."""

import pytest
import json
from unittest.mock import Mock, AsyncMock
from httpx import Response
import respx

from slime.router.handlers.openai_chat_completion import ChatCompletionHandler
from tests.router.fixtures.sample_requests import (
    get_simple_chat_request,
    get_multi_message_request,
    get_request_with_tools,
)
from tests.router.fixtures.mock_responses import get_sglang_generate_success


class TestCacheModeSelection:
    """Test that correct mode (cached vs direct) is selected."""

    @pytest.mark.asyncio
    async def test_cache_available_uses_cached_mode(self, mock_router, mock_request):
        """Test that cached mode is used when cache is available."""
        mock_router._check_cache_availability.return_value = True
        handler = ChatCompletionHandler(mock_router)
        handler._handle_with_radix_cache = AsyncMock(
            return_value=Response(content=b'{"result": "success"}', status_code=200)
        )

        await handler.handle_request(mock_request)

        handler._handle_with_radix_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_unavailable_uses_direct_mode(self, mock_router, mock_request):
        """Test that direct proxy mode is used when cache is unavailable."""
        mock_router._check_cache_availability.return_value = False
        handler = ChatCompletionHandler(mock_router)
        handler._proxy_to_sglang_chat_from_data = AsyncMock(
            return_value=Response(content=b'{"result": "success"}', status_code=200)
        )

        await handler.handle_request(mock_request)

        handler._proxy_to_sglang_chat_from_data.assert_called_once()


class TestRadixCacheRetrieval:
    """Test token retrieval from radix cache."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_retrieve_tokens_from_cache(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test that tokens are correctly retrieved from radix cache."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        # Setup mock responses
        expected_tokens = [1, 2, 3, 4, 5]
        expected_logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5]
        expected_loss_mask = [0, 0, 0, 0, 0]

        mock_radix_tree.retrieve_from_text = Mock(
            return_value=(expected_tokens, expected_logprobs, expected_loss_mask)
        )

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        await handler._handle_with_radix_cache(request_data)

        # Verify retrieve_from_text was called
        mock_radix_tree.retrieve_from_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_retrieval_failure_fallback(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test fallback to direct mode when cache retrieval fails."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        # Simulate cache retrieval failure
        mock_tokenizer.apply_chat_template.side_effect = RuntimeError("Template error")

        handler = ChatCompletionHandler(mock_router)
        handler._proxy_to_sglang_chat_from_data = AsyncMock(
            return_value=Response(content=b'{"result": "success"}', status_code=200)
        )

        request_data = get_simple_chat_request()
        await handler._handle_with_radix_cache(request_data)

        # Should fallback to direct proxy
        handler._proxy_to_sglang_chat_from_data.assert_called_once()


class TestRadixCacheInsertion:
    """Test token insertion into radix cache after generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_insert_tokens_after_generation(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test that generated tokens are inserted into cache."""
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

        await handler._handle_with_radix_cache(request_data)

        # Verify insert was called
        mock_radix_tree.insert.assert_called_once()

        # Verify insert was called with correct structure
        call_args = mock_radix_tree.insert.call_args[0]
        assert len(call_args) >= 2  # At least text and token_ids
        # text, token_ids are the first two arguments
        full_text = call_args[0]
        full_token_ids = call_args[1]
        assert isinstance(full_text, str)
        assert isinstance(full_token_ids, list)

    @pytest.mark.asyncio
    @respx.mock
    async def test_insert_failure_does_not_affect_response(self, mock_router, mock_radix_tree, mock_tokenizer):
        """Test that insert failure doesn't prevent response."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]
        mock_router.args.verbose = True

        # Simulate insert failure
        mock_radix_tree.insert.side_effect = RuntimeError("Cache insert error")

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        # Should not raise despite insert failure
        response = await handler._handle_with_radix_cache(request_data)
        assert response.status_code == 200


class TestMultipleRolloutsCacheStorage:
    """Test that multiple rollouts correctly store and retrieve tokens."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_multiple_rollouts_insert_called_multiple_times(
        self, mock_router, mock_radix_tree, mock_tokenizer
    ):
        """Test that insert is called for each generation rollout."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data1 = get_simple_chat_request()
        request_data2 = get_multi_message_request()

        # First rollout
        await handler._handle_with_radix_cache(request_data1)
        assert mock_radix_tree.insert.call_count == 1

        # Second rollout
        await handler._handle_with_radix_cache(request_data2)
        assert mock_radix_tree.insert.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_multiple_rollouts_logprobs_match_tokens(
        self, mock_router, mock_radix_tree, mock_tokenizer
    ):
        """Test that logprobs length matches token_ids length for each rollout."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        # Different response for each rollout
        mock_generate_response1 = {
            "text": "response1",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 10], [-0.2, 20], [-0.3, 30]],
                "finish_reason": {"type": "stop"}
            }
        }
        mock_generate_response2 = {
            "text": "response2",
            "meta_info": {
                "output_token_logprobs": [[-0.4, 40], [-0.5, 50]],
                "finish_reason": {"type": "stop"}
            }
        }

        # Mock different responses for each call
        respx.post("http://localhost:30000/generate").mock(
            side_effect=[
                Response(200, json=mock_generate_response1),
                Response(200, json=mock_generate_response2),
            ]
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        # First rollout
        await handler._handle_with_radix_cache(request_data)
        first_call_args = mock_radix_tree.insert.call_args_list[0][0]
        first_token_ids = first_call_args[1]
        first_logprobs = first_call_args[2]
        assert len(first_token_ids) == len(first_logprobs), \
            "First rollout: token_ids and logprobs length must match"

        # Second rollout
        await handler._handle_with_radix_cache(request_data)
        second_call_args = mock_radix_tree.insert.call_args_list[1][0]
        second_token_ids = second_call_args[1]
        second_logprobs = second_call_args[2]
        assert len(second_token_ids) == len(second_logprobs), \
            "Second rollout: token_ids and logprobs length must match"

    @pytest.mark.asyncio
    @respx.mock
    async def test_cached_tokens_reused_in_subsequent_requests(
        self, mock_router, mock_radix_tree, mock_tokenizer
    ):
        """Test that subsequent requests can retrieve previously cached tokens."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        # First request: cache miss, generates new tokens
        first_retrieve_tokens = [1, 2, 3]
        # Second request: cache hit, retrieves full cached sequence
        cached_tokens = first_retrieve_tokens + [10, 20, 30]  # Include output tokens
        cached_logprobs = [-0.1] * len(cached_tokens)

        # Use side_effect to return different values on successive calls
        mock_radix_tree.retrieve_from_text = Mock(
            side_effect=[
                (first_retrieve_tokens, [0.0, 0.0, 0.0], [0, 0, 0]),  # First call
                (cached_tokens, cached_logprobs, [0] * len(cached_tokens))  # Second call
            ]
        )

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        # First generation
        await handler._handle_with_radix_cache(request_data)

        # Second generation with same input - should retrieve cached tokens
        await handler._handle_with_radix_cache(request_data)

        # Verify retrieve was called twice
        assert mock_radix_tree.retrieve_from_text.call_count == 2


class TestChatTemplateWithTools:
    """Test that chat template correctly handles tools parameter."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_apply_chat_template_with_tools(
        self, mock_router, mock_radix_tree, mock_tokenizer
    ):
        """Test that tools are passed to apply_chat_template."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        mock_generate_response = get_sglang_generate_success()
        respx.post("http://localhost:30000/generate").mock(
            return_value=Response(200, json=mock_generate_response)
        )

        handler = ChatCompletionHandler(mock_router)
        request_data = get_request_with_tools()

        await handler._handle_with_radix_cache(request_data)

        # Verify apply_chat_template was called with tools
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_apply_chat_template_without_tools(
        self, mock_router, mock_radix_tree, mock_tokenizer
    ):
        """Test that None is passed when no tools in request."""
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

        await handler._handle_with_radix_cache(request_data)

        # Verify apply_chat_template was called with tools=None
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is None


class TestCacheFallbackScenarios:
    """Test various cache fallback scenarios."""

    @pytest.mark.asyncio
    async def test_empty_template_text_fallback(
        self, mock_router, mock_radix_tree, mock_tokenizer
    ):
        """Test fallback when template produces empty text."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]

        # Empty template result
        mock_tokenizer.apply_chat_template.return_value = ""

        handler = ChatCompletionHandler(mock_router)
        handler._proxy_to_sglang_chat_from_data = AsyncMock(
            return_value=Response(content=b'{"result": "success"}', status_code=200)
        )

        request_data = get_simple_chat_request()
        await handler._handle_with_radix_cache(request_data)

        # Should fallback to direct proxy
        handler._proxy_to_sglang_chat_from_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_tokenization_failure_fallback(
        self, mock_router, mock_radix_tree, mock_tokenizer
    ):
        """Test fallback when tokenization returns empty tokens."""
        mock_router.component_registry.get.side_effect = lambda x: {
            "radix_tree": mock_radix_tree,
            "tokenizer": mock_tokenizer
        }[x]
        mock_router.args.verbose = True

        # Empty tokens from cache
        mock_radix_tree.retrieve_from_text.return_value = ([], [], [])

        handler = ChatCompletionHandler(mock_router)
        handler._proxy_to_sglang_chat_from_data = AsyncMock(
            return_value=Response(content=b'{"result": "success"}', status_code=200)
        )

        request_data = get_simple_chat_request()
        await handler._handle_with_radix_cache(request_data)

        # Should fallback to direct proxy
        handler._proxy_to_sglang_chat_from_data.assert_called_once()
