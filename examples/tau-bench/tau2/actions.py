"""Canonical action format for tau2-bench training data and rollouts.

Supports two formats:

1. Native FC (Qwen3 compatible):
   <tool_call>
   {"name": "function_name", "arguments": {"arg1": "...", "arg2": "..."}}
   </tool_call>

2. Legacy format:
   <think>...</think>
   [ACTION]
   function_name(arg1="...", arg2="...")
   [/ACTION]

The parsed action is then mapped to tau2 gym `AgentGymEnv.step(action: str)`:
  - respond(content="...")  -> plain text message to the user
  - done()                  -> tool call that ends the episode
  - tool_name(...)          -> functional tool call

Parsing is lenient by default and auto-detects format.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParsedAction:
    raw_action_call: str
    name: str
    arguments: dict[str, Any]
    think: str


def _safe_eval_ast(node: Any) -> Any:
    import ast

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        return node.id
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.List):
        return [_safe_eval_ast(x) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_ast(x) for x in node.elts)
    if isinstance(node, ast.Dict):
        keys = [_safe_eval_ast(k) for k in node.keys]
        vals = [_safe_eval_ast(v) for v in node.values]
        return dict(zip(keys, vals, strict=True))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def parse_function_call(expr: str) -> tuple[str, dict[str, Any]]:
    import ast

    expr = expr.strip()
    tree = ast.parse(expr)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
        raise ValueError("Action must be a single function call expression.")
    call = tree.body[0].value
    if not isinstance(call.func, ast.Name):
        raise ValueError("Action must be a simple function call like foo(a=1).")
    name = call.func.id
    if call.args:
        raise ValueError("Positional arguments are not allowed; use keyword args only.")
    kwargs: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError("**kwargs is not supported.")
        kwargs[kw.arg] = _safe_eval_ast(kw.value)
    return name, kwargs


def parse_native_fc(text: str) -> ParsedAction | None:
    """Parse native function calling format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>

    Returns None if the format is not detected.
    """
    # Match <tool_call>...</tool_call>
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    name = data.get("name", "")
    arguments = data.get("arguments", {})

    if not name:
        return None

    # Handle string arguments (some models double-encode)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}

    # Extract think block if present (for thinking models)
    think = ""
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not think_match:
        think_match = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()

    # Build raw action call for compatibility
    args_str = ", ".join(f'{k}={json.dumps(v)}' for k, v in arguments.items())
    raw_action_call = f"{name}({args_str})"

    return ParsedAction(raw_action_call=raw_action_call, name=name, arguments=arguments, think=think)


def parse_action(text: str, *, require_think: bool = False) -> ParsedAction:
    """Parse a model response into a single action.

    Auto-detects format:
    1. Native FC: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. Legacy: [ACTION]function_name(arg="value")[/ACTION]

    Args:
        text: The raw model output
        require_think: If True, raise ValueError if think block is missing.
                      Default is False (lenient parsing for eval and diverse data).

    Accepts both <think>...</think> and <thinking>...</thinking> (Qwen3 native).
    """
    # Try native FC format first
    native_result = parse_native_fc(text)
    if native_result is not None:
        if require_think and not native_result.think:
            raise ValueError("Missing <think> or <thinking> block.")
        return native_result

    # Fall back to legacy [ACTION] format
    # Try <think> first, then <thinking> (Qwen3 native format)
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not think_match:
        think_match = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""

    if require_think and not think_match:
        raise ValueError("Missing <think> or <thinking> block.")

    if "[ACTION]" not in text or "[/ACTION]" not in text:
        raise ValueError("Missing [ACTION] or <tool_call> block.")

    start = text.index("[ACTION]") + len("[ACTION]")
    end = text.index("[/ACTION]", start)
    action_block = text[start:end].strip()
    if "\n" in action_block:
        lines = [ln.strip() for ln in action_block.splitlines() if ln.strip()]
        if len(lines) != 1:
            raise ValueError("Action block must contain exactly one non-empty line.")
        action_call = lines[0]
    else:
        action_call = action_block

    name, arguments = parse_function_call(action_call)
    return ParsedAction(raw_action_call=action_call, name=name, arguments=arguments, think=think)


def tool_result_block(tool_call: str, tool_output: str, *, native_fc: bool = False) -> str:
    """Format tool result for the next turn.

    Args:
        tool_call: The tool call string (e.g., "get_user(id=123)")
        tool_output: The tool's output
        native_fc: If True, use native FC format; otherwise use legacy format
    """
    tool_output = tool_output.strip()
    if native_fc:
        return f"Tool result for {tool_call}:\n{tool_output}"
    return f"[TOOL_RESULT]\n{tool_call}\n{tool_output}\n[/TOOL_RESULT]"


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> and <thinking>...</thinking> blocks from text.

    Used to clean teacher data for non-thinking student models.
    """
    # Remove </think> that appears without opening tag (thinking models default)
    text = re.sub(r"^\s*</think>\s*", "", text)
    # Remove full think blocks
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def format_as_native_fc(name: str, arguments: dict[str, Any]) -> str:
    """Format an action as native FC string.

    Args:
        name: Tool/function name
        arguments: Arguments dict

    Returns:
        String like: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    data = {"name": name, "arguments": arguments}
    return f"<tool_call>\n{json.dumps(data, indent=2)}\n</tool_call>"


def convert_legacy_to_native_fc(text: str, *, strip_think: bool = True) -> str | None:
    """Convert legacy [ACTION] format to native FC format.

    Args:
        text: Text possibly containing legacy [ACTION] block
        strip_think: If True, remove think blocks from output

    Returns:
        Converted text with native FC format, or None if conversion failed
    """
    try:
        parsed = parse_action(text)
    except (ValueError, SyntaxError):
        return None

    native_fc = format_as_native_fc(parsed.name, parsed.arguments)

    if strip_think:
        return native_fc
    elif parsed.think:
        return f"<think>\n{parsed.think}\n</think>\n{native_fc}"
    else:
        return native_fc


def strip_role_prefix(line: str) -> tuple[str | None, str]:
    line = line.strip()
    if ": " not in line:
        return None, line
    role, content = line.split(": ", 1)
    role = role.strip().lower()
    return role, content


@dataclass(frozen=True, slots=True)
class ParsedObservation:
    user_lines: tuple[str, ...]
    tool_lines: tuple[str, ...]
    assistant_lines: tuple[str, ...]
    other_lines: tuple[str, ...]

    @property
    def user_text(self) -> str:
        return "\n".join(self.user_lines).strip()

    @property
    def tool_text(self) -> str:
        return "\n".join(self.tool_lines).strip()

    @property
    def assistant_text(self) -> str:
        return "\n".join(self.assistant_lines).strip()

    @property
    def other_text(self) -> str:
        return "\n".join(self.other_lines).strip()

    @property
    def all_text(self) -> str:
        parts = [self.tool_text, self.assistant_text, self.user_text, self.other_text]
        return "\n".join([p for p in parts if p]).strip()


def split_observation(observation: str) -> ParsedObservation:
    """Parse a tau2 gym observation string into role-bucketed lines.

    AgentGymEnv emits newline-delimited lines like:
      "user: ...", "assistant: ...", "tool: ..."
    """
    user_lines: list[str] = []
    tool_lines: list[str] = []
    assistant_lines: list[str] = []
    other_lines: list[str] = []

    for raw_line in (observation or "").splitlines():
        role, content = strip_role_prefix(raw_line)
        content = content.strip()
        if not content:
            continue
        if role == "user":
            user_lines.append(content)
        elif role == "tool":
            tool_lines.append(content)
        elif role == "assistant":
            assistant_lines.append(content)
        else:
            # Preserve unknown roles and un-prefixed lines as-is.
            if role is None:
                other_lines.append(content)
            else:
                other_lines.append(f"{role}: {content}")

    return ParsedObservation(
        user_lines=tuple(user_lines),
        tool_lines=tuple(tool_lines),
        assistant_lines=tuple(assistant_lines),
        other_lines=tuple(other_lines),
    )


def observation_to_user_text(observation: str) -> str:
    """Convert tau2 gym observation string into plain text for the next model input."""
    if not observation:
        return ""
    parts: list[str] = []
    for raw_line in observation.splitlines():
        _, content = strip_role_prefix(raw_line)
        if content.strip():
            parts.append(content)
    return "\n".join(parts).strip()


def followup_messages_for_observation(
    *,
    observation: str,
    last_action_call: str,
    last_action_was_tool: bool,
    native_fc: bool = False,
) -> list[dict[str, str]]:
    """Build user-role messages for the next model turn.

    We keep the SFT/RL message schema minimal (system/user/assistant only) and
    represent tool outputs as a tagged user message.

    Args:
        observation: Raw observation string from tau2 gym
        last_action_call: The action call string from previous turn
        last_action_was_tool: Whether the previous action was a tool call
        native_fc: If True, use native FC format for tool results
    """
    parsed = split_observation(observation)
    messages: list[dict[str, str]] = []

    if last_action_was_tool:
        tool_payload = parsed.tool_text or parsed.all_text or "[no_observation]"
        messages.append({"role": "user", "content": tool_result_block(last_action_call, tool_payload, native_fc=native_fc)})
        if parsed.user_text:
            messages.append({"role": "user", "content": parsed.user_text})
        return messages

    user_payload = parsed.user_text or parsed.all_text or "[no_observation]"
    messages.append({"role": "user", "content": user_payload})
    return messages


def env_action_from_parsed_action(action: ParsedAction) -> str:
    """Convert a parsed action into the string passed to `AgentGymEnv.step`."""
    if action.name == "respond":
        content = action.arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("respond(content=...) requires a non-empty string content.")
        return content
    return action.raw_action_call


# ---------------------------------------------------------------------------
# Quality metrics and dedupe helpers
# ---------------------------------------------------------------------------


def reasoning_word_count(text: str) -> int:
    """Count words in a think block for quality filtering."""
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not think_match:
        think_match = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL)
    if not think_match:
        return 0
    content = think_match.group(1).strip()
    return len([w for w in content.split() if w])


def extract_tool_sequence_hash(trajectory: dict) -> str:
    """Hash of ordered tool names (exclude respond/done) for dedupe.

    Args:
        trajectory: Dict with "messages" key containing conversation turns.

    Returns:
        12-char hex hash of the tool sequence, or empty string if no tools.
    """
    tool_names: list[str] = []
    messages = trajectory.get("messages", [])

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        try:
            action = parse_action(content, require_think=False)
            if action.name not in ("respond", "done"):
                tool_names.append(action.name)
        except (ValueError, SyntaxError):
            # Skip unparseable messages
            continue

    if not tool_names:
        return ""
    return hashlib.md5("|".join(tool_names).encode()).hexdigest()[:12]


def canonicalize_think_block(text: str) -> str:
    """Convert <thinking> to <think> for canonical output format.

    If no think block exists, returns text unchanged (caller should add empty block if needed).
    """
    # Replace <thinking>...</thinking> with <think>...</think>
    return re.sub(
        r"<thinking>(.*?)</thinking>",
        r"<think>\1</think>",
        text,
        flags=re.DOTALL,
    )
