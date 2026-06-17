"""Model-output parsing helpers for agent harnesses."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ParsedModelOutput:
    """Structured view of one decoded model output."""

    reasoning: str
    text: str
    tool_uses: list[dict[str, Any]]


def parse_model_output(
    raw_output: str,
    *,
    tools_schema: list[dict] | None,
    tool_parser_name: str | None,
    reasoning_parser_name: str | None,
) -> ParsedModelOutput:
    """Parse raw model text into reasoning, visible text, and tool uses.

    The heavy format-specific work is delegated to SGLang's reasoning and
    function-call parsers. The XML fallback covers Anthropic-style tool-call
    text that some coding-agent models still emit occasionally.
    """
    reasoning, body_text = "", raw_output
    if reasoning_parser_name:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        r, b = ReasoningParser(
            model_type=reasoning_parser_name,
            stream_reasoning=False,
        ).parse_non_stream(raw_output)
        reasoning, body_text = r or "", b or ""
        if not reasoning and "</think>" in body_text:
            reasoning, body_text = body_text.split("</think>", 1)

    body_text, tool_uses = parse_tool_uses(body_text, tools_schema, tool_parser_name)
    if not tool_uses and tools_schema and reasoning and not body_text.strip():
        cleaned_reasoning, tool_uses = parse_gemma_native_tool_uses(reasoning, tools_schema)
        if tool_uses:
            reasoning = cleaned_reasoning.rstrip()
    return ParsedModelOutput(
        reasoning=reasoning,
        text=(body_text or "").strip(),
        tool_uses=tool_uses,
    )


def parse_tool_uses(
    body_text: str,
    tools_schema: list[dict] | None,
    tool_parser_name: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Parse tool calls from body text and return visible text plus tool uses."""
    tool_uses: list[dict[str, Any]] = []
    if tool_parser_name and tools_schema:
        from sglang.srt.entrypoints.openai.protocol import Function, Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        sg_tools = [Tool(type="function", function=Function(**d["function"])) for d in tools_schema]
        parser = FunctionCallParser(tools=sg_tools, tool_call_parser=tool_parser_name)
        calls = []
        if parser.has_tool_call(body_text):
            try:
                body_text, calls = parser.parse_non_stream(body_text)
            except Exception:
                logger.exception("[agent.parsing] sglang tool-call parsing failed; falling back")
        for c in calls:
            try:
                args = json.loads(c.parameters or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": c.parameters}
            tool_uses.append({"name": c.name or "tool", "input": args})

    if not tool_uses and tools_schema:
        body_text, tool_uses = parse_gemma_native_tool_uses(body_text, tools_schema)

    if not tool_uses and tools_schema:
        body_text, tool_uses = parse_xml_tool_uses(body_text, tools_schema)

    return body_text, tool_uses


_GEMMA_CALL_START = re.compile(r"call:([A-Za-z_][\w.-]*)\s*\{", flags=re.DOTALL)
_GEMMA_ARG_KEY = re.compile(r"\s*([A-Za-z_][\w.-]*|\"(?:\\.|[^\"])*\")\s*:", flags=re.DOTALL)


def parse_gemma_native_tool_uses(body_text: str, tools_schema: list[dict]) -> tuple[str, list[dict[str, Any]]]:
    """Fallback parser for Gemma-native trailing ``call:tool{...}`` calls."""
    valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
    match = _GEMMA_CALL_START.search(body_text)
    if match is None:
        return body_text, []
    parsed = _parse_gemma_native_tail(body_text, match.start(), valid_tools)
    if parsed is not None:
        return body_text[: match.start()], parsed
    return body_text, []


def _parse_gemma_native_tail(
    body_text: str,
    start: int,
    valid_tools: set[str | None],
) -> list[dict[str, Any]] | None:
    tool_uses: list[dict[str, Any]] = []
    pos = start
    while pos < len(body_text):
        pos = _skip_ws(body_text, pos)
        match = _GEMMA_CALL_START.match(body_text, pos)
        if match is None:
            return tool_uses if tool_uses and not body_text[pos:].strip() else None

        name = match.group(1)
        if name not in valid_tools:
            return None
        open_brace = match.end() - 1
        close_brace = _find_matching_brace(body_text, open_brace)
        if close_brace is None:
            return None
        args = _parse_gemma_native_args(body_text[open_brace + 1 : close_brace])
        if args is None:
            return None
        tool_uses.append({"name": name, "input": args})
        pos = close_brace + 1

    return tool_uses


def _find_matching_brace(text: str, open_brace: int) -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    pos = open_brace
    while pos < len(text):
        if quote is not None:
            ch = text[pos]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            pos += 1
            continue

        if text.startswith('<|"', pos):
            end = _find_gemma_string_end(text, pos)
            if end is None:
                return None
            pos = end
            continue

        ch = text[pos]
        if ch in {'"', "'"}:
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return pos
        pos += 1

    return None


def _find_gemma_string_end(text: str, start: int) -> int | None:
    pos = start + 3
    escaped = False
    while pos < len(text):
        ch = text[pos]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif text.startswith('"|>', pos):
            return pos + 3
        pos += 1
    return None


def _parse_gemma_native_args(args_text: str) -> dict[str, Any] | None:
    stripped = args_text.strip()
    if not stripped:
        return {}
    parsed = _parse_json_object_subset(stripped)
    if parsed is not None:
        return parsed
    return _parse_native_key_values(stripped)


def _parse_json_object_subset(args_text: str) -> dict[str, Any] | None:
    candidates = [args_text]
    if not args_text.startswith("{"):
        candidates.append(f"{{{args_text}}}")
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_native_key_values(args_text: str) -> dict[str, Any] | None:
    args: dict[str, Any] = {}
    pos = 0
    while pos < len(args_text):
        pos = _skip_ws(args_text, pos)
        if pos < len(args_text) and args_text[pos] == ",":
            pos = _skip_ws(args_text, pos + 1)
        if pos >= len(args_text):
            break

        key_match = _GEMMA_ARG_KEY.match(args_text, pos)
        if key_match is None:
            return None
        key = _decode_gemma_key(key_match.group(1))
        pos = key_match.end()
        value, pos = _parse_native_value(args_text, pos)
        args[key] = value

    return args


def _decode_gemma_key(raw_key: str) -> str:
    if raw_key.startswith('"'):
        try:
            decoded = json.loads(raw_key)
        except json.JSONDecodeError:
            return raw_key.strip('"')
        return str(decoded)
    return raw_key


def _parse_native_value(args_text: str, start: int) -> tuple[Any, int]:
    pos = _skip_ws(args_text, start)
    if args_text.startswith('<|"', pos):
        end = _find_gemma_string_end(args_text, pos)
        if end is not None:
            return _decode_gemma_string(args_text[pos + 3 : end - 3]), end

    parsed = _raw_decode_json_value(args_text, pos)
    if parsed is not None:
        value, end = parsed
        return value, end

    end = _find_native_value_end(args_text, pos)
    return args_text[pos:end].strip(), end


def _decode_gemma_string(raw_value: str) -> str:
    try:
        return str(json.loads(f'"{raw_value}"'))
    except json.JSONDecodeError:
        return raw_value


def _raw_decode_json_value(args_text: str, pos: int) -> tuple[Any, int] | None:
    if pos >= len(args_text) or args_text[pos] not in '"{[-0123456789tfn':
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(args_text[pos:])
    except json.JSONDecodeError:
        return None
    return value, pos + end


def _find_native_value_end(args_text: str, start: int) -> int:
    pos = start
    quote: str | None = None
    escaped = False
    while pos < len(args_text):
        if quote is not None:
            ch = args_text[pos]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
        elif args_text.startswith('<|"', pos):
            end = _find_gemma_string_end(args_text, pos)
            if end is None:
                return len(args_text)
            pos = end - 1
        elif args_text[pos] in {'"', "'"}:
            quote = args_text[pos]
        elif args_text[pos] == "," and _next_native_key_pos(args_text, pos + 1) is not None:
            return pos
        pos += 1
    return len(args_text)


def _next_native_key_pos(args_text: str, start: int) -> int | None:
    pos = _skip_ws(args_text, start)
    return pos if _GEMMA_ARG_KEY.match(args_text, pos) else None


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def parse_xml_tool_uses(body_text: str, tools_schema: list[dict]) -> tuple[str, list[dict[str, Any]]]:
    """Fallback parser for Anthropic-style XML tool calls."""
    valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
    tool_uses: list[dict[str, Any]] = []
    cleaned_parts: list[str] = []
    last = 0
    for m in re.finditer(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        body_text,
        flags=re.DOTALL,
    ):
        name, inner = m.group(1), m.group(2)
        if name in valid_tools:
            args = {
                p.group(1): p.group(2).strip()
                for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", inner, flags=re.DOTALL)
            }
            tool_uses.append({"name": name, "input": args})
            cleaned_parts.append(body_text[last : m.start()])
            last = m.end()
    cleaned_parts.append(body_text[last:])
    return "".join(cleaned_parts), tool_uses
