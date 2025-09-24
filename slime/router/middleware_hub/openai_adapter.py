from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool


def parse_tools(response: str, tools: List[Dict[str, Any]], parser: str = "qwen25"):
    """
    This function mimics the function call parser API from
    https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py#L952
    But running locally
    """
    tools_list = [
        Tool(
            function=Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool["type"],
        )
        for tool in tools
    ]
    parser = FunctionCallParser(tools=tools_list, tool_call_parser=parser)

    normal_text, calls = parser.parse_non_stream(response)

    return {
        "normal_text": normal_text,
        "calls": [call.model_dump() for call in calls],  # Convert pydantic objects to dictionaries
    }


@dataclass
class OpenAIToolCall:
    """OpenAI format tool call structure"""

    id: str
    type: str = "function"
    function: Dict[str, Any] = None


@dataclass
class OpenAIAssistantMessage:
    """OpenAI format assistant message structure"""

    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None


class OpenAICompatibleToolCallAdapter:
    """
    Adapter class that converts sglang tool call parsing results to OpenAI compatible format

    This class encapsulates existing tool call parsing and action conversion logic,
    and provides OpenAI format output interface.
    """

    def __init__(self, tools_info: List[Dict[str, Any]], parser_type: str = "qwen25"):
        """
        Initialize adapter

        Args:
            tools_info: List of tool information
            parser_type: Parser type, defaults to "qwen25"
        """
        self.tools_info = tools_info
        self.parser_type = parser_type

    def parse_response_to_openai_format(self, response: str) -> Dict[str, Any]:
        """
        Parse sglang response to OpenAI compatible format

        Args:
            response: Raw response text from sglang

        Returns:
            Dictionary containing OpenAI format message and parsing results

        Raises:
            Exception: Thrown when parsing fails
        """
        print(f"[OpenAI Adapter] Starting to parse response: {response[:100]}...")

        try:
            # Use existing parser to parse tool calls
            print(f"[OpenAI Adapter] Using parser type: {self.parser_type}")
            parsed = parse_tools(response, self.tools_info, self.parser_type)
            print(f"[OpenAI Adapter] Parsing successful. Normal text: '{parsed['normal_text']}'")
            print(f"[OpenAI Adapter] Found {len(parsed['calls'])} tool calls: {parsed['calls']}")

            # Extract parsing results
            normal_text = parsed["normal_text"]
            calls = parsed["calls"]

            # Convert to OpenAI format
            openai_message = self._convert_to_openai_message(normal_text, calls)
            print(f"[OpenAI Adapter] OpenAI message created: {openai_message}")

            return {"openai_message": openai_message, "parsed_result": parsed, "success": True}

        except Exception as e:
            print(f"[OpenAI Adapter] Parsing failed with error: {str(e)}")
            return {"openai_message": None, "parsed_result": None, "success": False, "error": str(e)}

    def _convert_to_openai_message(self, normal_text: str, calls: List[Dict[str, Any]]) -> OpenAIAssistantMessage:
        """
        Convert parsing results to OpenAI format assistant message

        Args:
            normal_text: Normal text content
            calls: List of tool calls

        Returns:
            OpenAI format assistant message
        """
        print(f"[OpenAI Adapter] Converting to OpenAI format - normal_text: '{normal_text}', calls: {calls}")

        if not calls:
            # No tool calls, return plain text response
            print("[OpenAI Adapter] No tool calls found, returning plain text response")
            return OpenAIAssistantMessage(role="assistant", content=normal_text, tool_calls=None)

        # Convert tool calls to OpenAI format
        print(f"[OpenAI Adapter] Converting {len(calls)} tool calls to OpenAI format")
        openai_tool_calls = []
        for i, call in enumerate(calls):
            print(f"[OpenAI Adapter] Processing call {i}: {call}")
            openai_tool_call = OpenAIToolCall(
                id=f"call_{i}_{call.get('name', 'unknown')}",
                type="function",
                function={"name": call.get("name", ""), "arguments": call.get("parameters", "{}")},
            )
            print(f"[OpenAI Adapter] Created OpenAI tool call: {openai_tool_call}")
            openai_tool_calls.append(openai_tool_call)

        result = OpenAIAssistantMessage(
            role="assistant", content=normal_text if normal_text.strip() else None, tool_calls=openai_tool_calls
        )
        print(f"[OpenAI Adapter] Final OpenAI message: {result}")
        return result
