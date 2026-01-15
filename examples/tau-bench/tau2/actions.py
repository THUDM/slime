"""Tau2 action parsing and observation formatting.

We standardize on Qwen3 native function calling:

  <tool_call>{"name": "...", "arguments": {...}}</tool_call>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"


@dataclass(frozen=True, slots=True)
class ParsedAction:
    name: str
    arguments: dict[str, Any]
    raw_action_call: str


def _to_py_literal(value: Any) -> str:
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_to_py_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            items.append(f"{_to_py_literal(k)}: {_to_py_literal(v)}")
        return "{" + ", ".join(items) + "}"
    return repr(value)


def _to_functional_call(name: str, arguments: dict[str, Any]) -> str:
    if not arguments:
        return f"{name}()"
    parts = [f"{k}={_to_py_literal(v)}" for k, v in arguments.items()]
    return f"{name}({', '.join(parts)})"


def _find_tool_call_blocks(text: str) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    cursor = 0
    while True:
        start = text.find(_TOOL_CALL_START, cursor)
        if start == -1:
            break
        end = text.find(_TOOL_CALL_END, start + len(_TOOL_CALL_START))
        if end == -1:
            raise ValueError("Missing </tool_call> for <tool_call> block")
        content = text[start + len(_TOOL_CALL_START) : end].strip()
        blocks.append((content, start, end + len(_TOOL_CALL_END)))
        cursor = end + len(_TOOL_CALL_END)
    return blocks


def parse_action(text: str) -> ParsedAction:
    blocks = _find_tool_call_blocks(text)
    if not blocks:
        raise ValueError("Missing <tool_call>...</tool_call> block")
    if len(blocks) > 1:
        raise ValueError("Multiple <tool_call> blocks found; expected exactly one")

    content, start, end = blocks[0]
    prefix = text[:start].strip()
    suffix = text[end:].strip()
    if prefix or suffix:
        raise ValueError("Unexpected text outside <tool_call> block")

    data = json.loads(content)
    name = data.get("name")
    arguments = data.get("arguments") or {}

    if not isinstance(name, str) or not name:
        raise ValueError("Tool call missing non-empty 'name'")

    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    if not isinstance(arguments, dict):
        raise ValueError("Tool call 'arguments' must be an object")

    return ParsedAction(name=name, arguments=arguments, raw_action_call=_to_functional_call(name, arguments))


def _strip_role_prefix(line: str) -> tuple[str | None, str]:
    line = line.strip()
    if ": " not in line:
        return None, line
    role, content = line.split(": ", 1)
    return role.strip().lower(), content


@dataclass(frozen=True, slots=True)
class ParsedObservation:
    user: str
    tool: str
    other: str


def split_observation(observation: str) -> ParsedObservation:
    user_lines: list[str] = []
    tool_lines: list[str] = []
    other_lines: list[str] = []

    for raw_line in (observation or "").splitlines():
        role, content = _strip_role_prefix(raw_line)
        content = content.strip()
        if not content:
            continue
        if role == "user":
            user_lines.append(content)
        elif role == "tool":
            tool_lines.append(content)
        else:
            other_lines.append(content if role is None else f"{role}: {content}")

    return ParsedObservation(
        user="\n".join(user_lines).strip(),
        tool="\n".join(tool_lines).strip(),
        other="\n".join(other_lines).strip(),
    )


def followup_messages_for_observation(
    *,
    observation: str,
    last_action_call: str,
    last_action_was_tool: bool,
) -> list[dict[str, str]]:
    parsed = split_observation(observation)
    messages: list[dict[str, str]] = []

    if last_action_was_tool:
        tool_payload = parsed.tool or parsed.other or "[no_observation]"
        messages.append({"role": "user", "content": f"Tool result for {last_action_call}:\n{tool_payload}"})
        if parsed.user:
            messages.append({"role": "user", "content": parsed.user})
        return messages

    user_payload = parsed.user or parsed.other or "[no_observation]"
    messages.append({"role": "user", "content": user_payload})
    return messages


def env_action_from_parsed_action(action: ParsedAction) -> str:
    if action.name == "respond":
        content = action.arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("respond requires a non-empty content string")
        return content
    return action.raw_action_call
