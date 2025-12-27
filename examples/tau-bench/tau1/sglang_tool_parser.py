import logging
from typing import Any

logger = logging.getLogger(__name__)

_SGLANG_AVAILABLE = False
_FunctionCallParser = None
_Function = None
_Tool = None

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser as _FunctionCallParser
    from sglang.srt.managers.io_struct import Function as _Function
    from sglang.srt.managers.io_struct import Tool as _Tool

    _SGLANG_AVAILABLE = True
except Exception as exc:
    logger.warning(f"sglang tool parser unavailable (optional). Falling back to no-tool parsing. Error: {exc}")


def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = "qwen25") -> dict[str, Any]:
    if not _SGLANG_AVAILABLE:
        return {"normal_text": response, "calls": []}

    tools_list = [
        _Tool(
            function=_Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool.get("type", "function"),
        )
        for tool in tools
    ]

    parser_obj = _FunctionCallParser(tools=tools_list, tool_call_parser=parser)
    try:
        normal_text, calls = parser_obj.parse_non_stream(response)
    except Exception as exc:
        logger.warning(f"sglang tool parser failed, falling back to no-tool parsing. Error: {exc}")
        return {"normal_text": response, "calls": []}

    calls = [call.model_dump() if hasattr(call, "model_dump") else call for call in calls]
    return {"normal_text": normal_text, "calls": calls}
