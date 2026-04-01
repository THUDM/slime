from typing import Any

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool


def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = "qwen25"):
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
