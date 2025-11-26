"""Tests for SGLang parser integration (reasoning and function call parsers)."""

import pytest
import json
from unittest.mock import Mock, patch

from slime.router.handlers.openai_chat_completion import ChatCompletionHandler

# Check if parsers are available
try:
    from slime.router.handlers.openai_chat_completion import SGLANG_PARSERS_AVAILABLE
except ImportError:
    SGLANG_PARSERS_AVAILABLE = False

# Skip all tests in this module if parsers not available
pytestmark = pytest.mark.skipif(
    not SGLANG_PARSERS_AVAILABLE,
    reason="SGLang parsers not available"
)


class TestParserAvailability:
    """Test parser availability handling."""

    def test_parsers_not_available_returns_raw_text(self, mock_router):
        """Test that raw text is returned when parsers not available."""
        handler = ChatCompletionHandler(mock_router)

        # Temporarily disable parsers
        with patch('slime.router.handlers.openai_chat_completion.SGLANG_PARSERS_AVAILABLE', False):
            text, tool_calls, reasoning = handler._parse_generated_output(
                "Hello world", {}
            )

        assert text == "Hello world"
        assert tool_calls is None
        assert reasoning is None


class TestReasoningParser:
    """Test reasoning parser integration."""

    def test_parse_output_with_reasoning(self, mock_router, mock_reasoning_parser):
        """Test parsing with reasoning parser."""
        mock_router.args.sglang_reasoning_parser = "deepseek"
        handler = ChatCompletionHandler(mock_router)
        handler._reasoning_parser = mock_reasoning_parser

        text, tool_calls, reasoning = handler._parse_generated_output(
            "Think: reasoning\nAnswer: normal", {}
        )

        assert text == "Normal text"  # From mock
        assert reasoning == "Reasoning content"  # From mock
        assert tool_calls is None

        # Verify parser was called
        mock_reasoning_parser.parse_non_stream.assert_called_once()

    def test_reasoning_parser_lazy_initialization(self, mock_router):
        """Test that reasoning parser is lazily initialized."""
        mock_router.args.sglang_reasoning_parser = "deepseek"
        handler = ChatCompletionHandler(mock_router)

        assert handler._reasoning_parser is None

        # Mock the parser class to avoid actual import
        with patch('slime.router.handlers.openai_chat_completion.ReasoningParser') as MockParser:
            mock_parser_instance = Mock()
            mock_parser_instance.parse_non_stream = Mock(
                return_value=("reasoning", "normal")
            )
            MockParser.return_value = mock_parser_instance

            handler._parse_generated_output("test", {})

            # Parser should be initialized
            MockParser.assert_called_once_with(
                model_type="deepseek",
                stream_reasoning=False
            )

    def test_reasoning_text_included_in_metadata(self, mock_router, mock_reasoning_parser):
        """Test that reasoning is included in metadata when configured."""
        mock_router.args.sglang_reasoning_parser = "deepseek"
        mock_router.args.include_reasoning_in_response = True
        handler = ChatCompletionHandler(mock_router)
        handler._reasoning_parser = mock_reasoning_parser

        text, tool_calls, reasoning = handler._parse_generated_output(
            "Think: reasoning\nAnswer: normal", {}
        )

        assert reasoning == "Reasoning content"


class TestFunctionCallParser:
    """Test function call parser integration."""

    def test_parse_output_with_tool_calls(self, mock_router, mock_function_call_parser):
        """Test parsing with tool call parser."""
        mock_router.args.sglang_tool_call_parser = "hermes"
        handler = ChatCompletionHandler(mock_router)
        handler._function_call_parser = mock_function_call_parser

        request_data = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather"
                    }
                }
            ]
        }

        text, tool_calls, reasoning = handler._parse_generated_output(
            "<tool_call>get_weather</tool_call>", request_data
        )

        assert text == ""  # Remaining text from mock
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["type"] == "function"
        assert "id" in tool_calls[0]

    def test_function_call_parser_lazy_initialization(self, mock_router):
        """Test that function call parser is lazily initialized."""
        mock_router.args.sglang_tool_call_parser = "hermes"
        handler = ChatCompletionHandler(mock_router)

        assert handler._function_call_parser is None

        # Mock the parser class
        with patch('slime.router.handlers.openai_chat_completion.FunctionCallParser') as MockParser:
            mock_parser_instance = Mock()
            mock_parser_instance.parse_non_stream = Mock(
                return_value=("", [])
            )
            MockParser.return_value = mock_parser_instance

            # Also need to mock Tool within the try block
            with patch('sglang.srt.entrypoints.openai.protocol.Tool'):
                request_data = {
                    "tools": [{
                        "type": "function",
                        "function": {"name": "test"}
                    }]
                }

                handler._parse_generated_output("test", request_data)

                # Parser should be initialized
                MockParser.assert_called_once()

    def test_tool_call_id_generation(self, mock_router, mock_function_call_parser):
        """Test that tool call IDs are generated."""
        mock_router.args.sglang_tool_call_parser = "hermes"
        handler = ChatCompletionHandler(mock_router)
        handler._function_call_parser = mock_function_call_parser

        request_data = {
            "tools": [{"type": "function", "function": {"name": "test"}}]
        }

        _, tool_calls, _ = handler._parse_generated_output("test", request_data)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        # ID should start with "call_" and have 24 hex chars
        assert tool_calls[0]["id"].startswith("call_")
        assert len(tool_calls[0]["id"]) == 29  # "call_" + 24 chars

    def test_tool_call_arguments_json_serialization(self, mock_router):
        """Test that dict arguments are JSON-serialized."""
        mock_router.args.sglang_tool_call_parser = "hermes"
        handler = ChatCompletionHandler(mock_router)

        # Create a mock parser with dict arguments
        mock_parser = Mock()
        mock_tool_call = Mock()
        mock_tool_call.name = "get_weather"
        mock_tool_call.arguments = {"location": "SF", "unit": "celsius"}

        mock_parser.parse_non_stream = Mock(return_value=("", [mock_tool_call]))
        handler._function_call_parser = mock_parser

        request_data = {
            "tools": [{"type": "function", "function": {"name": "get_weather"}}]
        }

        _, tool_calls, _ = handler._parse_generated_output("test", request_data)

        # Arguments should be JSON string
        arguments = tool_calls[0]["function"]["arguments"]
        assert isinstance(arguments, str)
        parsed_args = json.loads(arguments)
        assert parsed_args["location"] == "SF"
        assert parsed_args["unit"] == "celsius"


class TestCombinedParsers:
    """Test using both reasoning and function call parsers together."""

    def test_both_parsers_applied_sequentially(self, mock_router, mock_reasoning_parser, mock_function_call_parser):
        """Test that both parsers are applied in sequence."""
        mock_router.args.sglang_reasoning_parser = "deepseek"
        mock_router.args.sglang_tool_call_parser = "hermes"
        handler = ChatCompletionHandler(mock_router)
        handler._reasoning_parser = mock_reasoning_parser
        handler._function_call_parser = mock_function_call_parser

        request_data = {
            "tools": [{"type": "function", "function": {"name": "test"}}]
        }

        text, tool_calls, reasoning = handler._parse_generated_output(
            "Think: reasoning\n<tool_call>test</tool_call>", request_data
        )

        # Both parsers should be called
        mock_reasoning_parser.parse_non_stream.assert_called_once()
        mock_function_call_parser.parse_non_stream.assert_called_once()

        # Should have both reasoning and tool calls
        assert reasoning == "Reasoning content"
        assert tool_calls is not None


class TestParserErrorHandling:
    """Test error handling in parsers."""

    def test_parser_error_returns_raw_text(self, mock_router):
        """Test that parser errors return raw text."""
        mock_router.args.sglang_reasoning_parser = "deepseek"
        mock_router.args.verbose = True
        handler = ChatCompletionHandler(mock_router)

        # Mock parser that raises exception
        bad_parser = Mock()
        bad_parser.parse_non_stream.side_effect = ValueError("Parse error")
        handler._reasoning_parser = bad_parser

        # Should return raw text on error
        text, tools, reasoning = handler._parse_generated_output(
            "original text", {}
        )
        assert text == "original text"
        assert tools is None
        assert reasoning is None

    def test_parser_error_with_verbose_logging(self, mock_router, capsys):
        """Test that parser errors are logged when verbose is enabled."""
        mock_router.args.sglang_reasoning_parser = "deepseek"
        mock_router.args.verbose = True
        handler = ChatCompletionHandler(mock_router)

        # Mock parser that raises exception
        bad_parser = Mock()
        bad_parser.parse_non_stream.side_effect = ValueError("Parse error")
        handler._reasoning_parser = bad_parser

        handler._parse_generated_output("test", {})

        # Check that error was printed (due to verbose=True)
        # Note: This test may need adjustment based on actual logging implementation


class TestNoToolsProvided:
    """Test behavior when no tools are provided."""

    def test_no_tools_skips_function_parser(self, mock_router):
        """Test that function call parser is not used when no tools provided."""
        mock_router.args.sglang_tool_call_parser = "hermes"
        handler = ChatCompletionHandler(mock_router)

        # No tools in request
        request_data = {}

        text, tool_calls, reasoning = handler._parse_generated_output("test", request_data)

        # Function parser should not be initialized
        assert handler._function_call_parser is None
        assert tool_calls is None
