"""Tests for chat completion request validation."""

import pytest
import json
from fastapi import HTTPException
from unittest.mock import Mock

from slime.router.handlers.openai_chat_completion import ChatCompletionHandler
from tests.router.fixtures.sample_requests import (
    get_simple_chat_request,
    get_invalid_request_no_messages,
    get_invalid_request_empty_messages,
    get_invalid_request_messages_not_list,
    get_request_with_null_content,
    get_streaming_request,
)


class TestChatCompletionValidation:
    """Test request validation for chat completion API."""

    def test_valid_request(self, mock_router):
        """Test validation passes for valid request."""
        handler = ChatCompletionHandler(mock_router)
        request_data = get_simple_chat_request()

        # Should not raise
        handler._validate_chat_completion_request(request_data)

    def test_missing_messages_field(self, mock_router):
        """Test validation fails when messages field is missing."""
        handler = ChatCompletionHandler(mock_router)
        request_data = get_invalid_request_no_messages()

        with pytest.raises(HTTPException) as exc_info:
            handler._validate_chat_completion_request(request_data)
        assert exc_info.value.status_code == 400
        assert "messages" in exc_info.value.detail.lower()

    def test_empty_messages_list(self, mock_router):
        """Test validation fails for empty messages list."""
        handler = ChatCompletionHandler(mock_router)
        request_data = get_invalid_request_empty_messages()

        with pytest.raises(HTTPException) as exc_info:
            handler._validate_chat_completion_request(request_data)
        assert exc_info.value.status_code == 400
        assert "non-empty" in exc_info.value.detail.lower()

    def test_messages_not_list(self, mock_router):
        """Test validation fails when messages is not a list."""
        handler = ChatCompletionHandler(mock_router)
        request_data = get_invalid_request_messages_not_list()

        with pytest.raises(HTTPException) as exc_info:
            handler._validate_chat_completion_request(request_data)
        assert exc_info.value.status_code == 400

    def test_message_not_dict(self, mock_router):
        """Test validation fails when message is not a dict."""
        handler = ChatCompletionHandler(mock_router)
        request_data = {"messages": ["not a dict"]}

        with pytest.raises(HTTPException) as exc_info:
            handler._validate_chat_completion_request(request_data)
        assert exc_info.value.status_code == 400
        assert "dictionary" in exc_info.value.detail.lower()

    def test_message_missing_role(self, mock_router):
        """Test validation fails when message missing role field."""
        handler = ChatCompletionHandler(mock_router)
        request_data = {"messages": [{"content": "Hello"}]}

        with pytest.raises(HTTPException) as exc_info:
            handler._validate_chat_completion_request(request_data)
        assert exc_info.value.status_code == 400
        assert "role" in exc_info.value.detail.lower()

    def test_message_null_content_normalized(self, mock_router):
        """Test that null/missing content is normalized to empty string."""
        handler = ChatCompletionHandler(mock_router)

        # Test null content
        request_data = {
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": []}
            ]
        }
        handler._validate_chat_completion_request(request_data)
        assert request_data["messages"][0]["content"] == ""

        # Test missing content
        request_data2 = {
            "messages": [
                {"role": "assistant", "tool_calls": []}
            ]
        }
        handler._validate_chat_completion_request(request_data2)
        assert request_data2["messages"][0]["content"] == ""

    def test_message_with_tools_null_content(self, mock_router):
        """Test that messages with tool_calls can have null content."""
        handler = ChatCompletionHandler(mock_router)
        request_data = get_request_with_null_content()

        # Should not raise - valid per OpenAI spec
        handler._validate_chat_completion_request(request_data)
        # Content should be normalized to empty string
        assert request_data["messages"][0]["content"] == ""


class TestStreamingRequestRejection:
    """Test that streaming requests are properly rejected."""

    @pytest.mark.asyncio
    async def test_streaming_request_rejected(self, mock_router):
        """Test that streaming requests are rejected with appropriate error."""
        async def json_with_stream():
            return get_streaming_request()

        request = Mock()
        request.json = json_with_stream

        handler = ChatCompletionHandler(mock_router)

        with pytest.raises(HTTPException) as exc_info:
            await handler.handle_request(request)
        assert exc_info.value.status_code == 400
        assert "streaming is not supported" in exc_info.value.detail.lower()


class TestInvalidJSONHandling:
    """Test handling of invalid JSON in requests."""

    @pytest.mark.asyncio
    async def test_handle_request_invalid_json(self, mock_router):
        """Test handle_request with invalid JSON."""
        async def bad_json():
            raise json.JSONDecodeError("test", "", 0)

        request = Mock()
        request.json = bad_json

        handler = ChatCompletionHandler(mock_router)

        with pytest.raises(HTTPException) as exc_info:
            await handler.handle_request(request)
        assert exc_info.value.status_code == 400
        assert "invalid json" in exc_info.value.detail.lower()
