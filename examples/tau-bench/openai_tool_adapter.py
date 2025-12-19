import json
import logging
from dataclasses import dataclass
from typing import Any

from sglang_tool_parser import parse_tools
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME
from tau_bench.types import Action

# Set up logger for this module
logger = logging.getLogger(__name__)


def _parse_tools_compat(parse_tools_fn, response: str, tools_info, parser_type: str | None):
    """
    Compatible wrapper for parse_tools() across versions.

    Some versions: parse_tools(response, tools_info)
    Some versions: parse_tools(response, tools_info, parser_type)
    """
    try:
        # try 3-arg signature first
        return parse_tools_fn(response, tools_info, parser_type)
    except TypeError:
        # fallback to 2-arg signature
        return parse_tools_fn(response, tools_info)


@dataclass
class OpenAIToolCall:
    """OpenAI format tool call structure"""

    id: str
    type: str = "function"
    function: dict[str, Any] = None


@dataclass
class OpenAIAssistantMessage:
    """OpenAI format assistant message structure"""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAICompatibleToolCallAdapter:
    """
    Adapter class that converts sglang tool call parsing results to OpenAI compatible format

    This class encapsulates existing tool call parsing and action conversion logic,
    and provides OpenAI format output interface.
    """

    def __init__(self, tools_info: list[dict[str, Any]], parser_type: str = "qwen25"):
        """
        Initialize adapter

        Args:
            tools_info: List of tool information
            parser_type: Parser type, defaults to "qwen25"
        """
        self.tools_info = tools_info
        self.parser_type = parser_type

    def parse_response_to_openai_format(self, response: str) -> dict[str, Any]:
        """
        Parse sglang response to OpenAI compatible format

        Args:
            response: Raw response text from sglang

        Returns:
            Dictionary containing OpenAI format message and parsing results

        Raises:
            Exception: Thrown when parsing fails
        """
        try:
            parsed = _parse_tools_compat(parse_tools, response, self.tools_info, self.parser_type)

            # --- normalize parsed output ---
            normal_text = None
            calls = []

            # Case A: dict output
            if isinstance(parsed, dict):
                # common keys
                if "normal_text" in parsed:
                    normal_text = parsed.get("normal_text")
                elif "text" in parsed:
                    normal_text = parsed.get("text")
                elif "content" in parsed:
                    normal_text = parsed.get("content")
                elif "assistant" in parsed:
                    normal_text = parsed.get("assistant")
                else:
                    # fallback: if no known key, just use raw response
                    normal_text = response

                # tool calls keys variants
                if "calls" in parsed and isinstance(parsed.get("calls"), list):
                    calls = parsed.get("calls", [])
                elif "tool_calls" in parsed and isinstance(parsed.get("tool_calls"), list):
                    calls = parsed.get("tool_calls", [])
                elif "function_calls" in parsed and isinstance(parsed.get("function_calls"), list):
                    calls = parsed.get("function_calls", [])
                else:
                    calls = []

            # Case B: tuple/list output (e.g., (normal_text, calls))
            elif isinstance(parsed, (list, tuple)):
                if len(parsed) >= 1:
                    normal_text = parsed[0]
                if len(parsed) >= 2 and isinstance(parsed[1], list):
                    calls = parsed[1]
                if normal_text is None:
                    normal_text = response

            # Case C: unknown type
            else:
                normal_text = response
                calls = []

            # Ensure types
            if normal_text is None:
                normal_text = response
            if not isinstance(calls, list):
                calls = []

            # Convert to OpenAI format
            openai_message = self._convert_to_openai_message(normal_text, calls)

            # For downstream code, we still provide parsed_result in the "tau expected" schema
            parsed_result = {"normal_text": normal_text, "calls": calls}

            return {"openai_message": openai_message, "parsed_result": parsed_result, "success": True}

        except Exception as e:
            logger.warning(f"Parsing failed with error: {type(e).__name__}: {str(e)}")
            return {"openai_message": None, "parsed_result": None, "success": False, "error": str(e)}

    def _convert_to_openai_message(self, normal_text: str, calls: list[dict[str, Any]]) -> OpenAIAssistantMessage:
        """
        Convert parsing results to OpenAI format assistant message

        Args:
            normal_text: Normal text content
            calls: List of tool calls

        Returns:
            OpenAI format assistant message
        """
        if not calls:
            # No tool calls, return plain text response
            return OpenAIAssistantMessage(role="assistant", content=normal_text, tool_calls=None)

        # Convert tool calls to OpenAI format
        openai_tool_calls = []
        for i, call in enumerate(calls):
            openai_tool_call = OpenAIToolCall(
                id=f"call_{i}_{call.get('name', 'unknown')}",
                type="function",
                function={"name": call.get("name", ""), "arguments": call.get("parameters", "{}")},
            )
            openai_tool_calls.append(openai_tool_call)

        result = OpenAIAssistantMessage(
            role="assistant", content=normal_text if normal_text.strip() else None, tool_calls=openai_tool_calls
        )
        return result

    def _call_to_action_sglang(self, calls: list[Any], text_response: str) -> Action:
        """
        Convert sglang tool calls to Action object

        This method replicates the original call_to_action_sglang logic,
        ensuring compatibility with existing code.
        """
        # Default action if no action found
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})

        if calls:
            if len(calls) > 1:
                logger.debug("Multiple tool calls identified, only taking first.")

            tool_call = calls[0]

            try:
                params = json.loads(tool_call["parameters"])

                if not isinstance(params, dict):
                    logger.warning(f"{params} does not follow dict structure for action")
                else:
                    action = Action(name=tool_call["name"], kwargs=params)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse parameters as JSON: {e}")

        return action

    def get_openai_tools_format(self) -> list[dict[str, Any]]:
        """
        Get OpenAI format tool definitions

        Returns:
            List of OpenAI format tools
        """
        openai_tools = []
        for tool in self.tools_info:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"],
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools


# Usage examples and factory functions
def create_openai_adapter(
    tools_info: list[dict[str, Any]], parser_type: str = "qwen25"
) -> OpenAICompatibleToolCallAdapter:
    """
    Factory function to create OpenAI compatible tool call adapter

    Args:
        tools_info: List of tool information
        parser_type: Parser type

    Returns:
        Configured adapter instance
    """
    return OpenAICompatibleToolCallAdapter(tools_info, parser_type)
