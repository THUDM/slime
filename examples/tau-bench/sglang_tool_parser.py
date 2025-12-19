import logging
from typing import Any

logger = logging.getLogger(__name__)

# We make sglang an OPTIONAL dependency.
# On many dev machines (e.g., macOS / CPU-only / Python 3.13), importing sglang.srt can
# pull in heavy deps like triton. For Tau-Bench integration we should not hard-require it.

_SGLANG_AVAILABLE = False
_FunctionCallParser = None
_Function = None
_Tool = None

try:
    # These imports may trigger triton dependency in some sglang versions.
    from sglang.srt.function_call.function_call_parser import FunctionCallParser as _FunctionCallParser
    from sglang.srt.managers.io_struct import Function as _Function
    from sglang.srt.managers.io_struct import Tool as _Tool

    _SGLANG_AVAILABLE = True
except Exception as e:
    logger.warning(f"sglang tool parser unavailable (optional). Falling back to no-tool parsing. Error: {e}")


def parse_tools(tools_info: list[dict[str, Any]], text: str) -> dict[str, Any]:
    """
    Parse tool calls from model output text.

    Returns a dict compatible with openai_tool_adapter expectations:
      {
        "success": bool,
        "error": str | None,
        "parsed_result": {"normal_text": str, "calls": list[dict]}
      }
    """
    # Fallback: treat everything as normal text; no tool calls.
    if not _SGLANG_AVAILABLE:
        return {
            "success": True,
            "error": None,
            "parsed_result": {"normal_text": text, "calls": []},
        }

    # If sglang parser is available, use it.
    try:
        tools = [_Tool(**t) for t in tools_info]
        functions = [_Function.from_tool(t) for t in tools]  # depending on sglang version
        parser = _FunctionCallParser(functions)

        normal_text, calls = parser.parse(text)
        # calls should be list of dict: [{"name":..., "parameters": "...json..."}]
        return {
            "success": True,
            "error": None,
            "parsed_result": {"normal_text": normal_text, "calls": calls or []},
        }
    except Exception as e:
        return {
            "success": False,
            "error": repr(e),
            "parsed_result": {"normal_text": text, "calls": []},
        }
