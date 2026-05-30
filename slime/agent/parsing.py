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
        body_text, tool_uses = parse_xml_tool_uses(body_text, tools_schema)

    return body_text, tool_uses


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
