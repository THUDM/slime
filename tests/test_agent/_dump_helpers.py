"""Tree dumper for the trajectory_manager branching test.

Renders ``TrajectoryManager._trees[sid]`` as ASCII text so a human can read the
routing tree next to the linearized Samples. Adapted to the refactored
``MessageNode`` API (``.message`` / ``.turn`` / ``.turn_index``); see
``slime/agent/trajectory.py``.

Pulled out of the historic ``tests/test_coding_agent/_dump_helpers.py`` (which
also carried JSON dumpers + an aiohttp debug middleware) down to just the one
helper the branching test imports.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def dump_tree_txt(manager, sid: str, *, max_text_chars: int = 80) -> str:
    """Render ``manager._trees[sid]`` as ASCII text.

    Returns ``"<no session: {sid}>"`` when the sid has no tree (already drained
    or never opened). Otherwise a header line plus one indented row per non-root
    node, listing role / message text and, for generated assistant leaves, the
    turn snapshot fields (turn index, prompt/response id counts, finish reason).
    """
    root = manager._trees.get(sid)
    if root is None:
        return f"<no session: {sid}>"
    non_root_count = sum(1 for _ in _iter_non_root(root))
    leaf_count = sum(1 for leaf in root.leaves() if not leaf.is_root)
    lines: list[str] = [
        f"session={sid} turns={manager.turn_count(sid)} leaves={leaf_count} nodes={non_root_count}",
        "root",
    ]
    _render_subtree(root, lines, depth=0, max_text_chars=max_text_chars)
    return "\n".join(lines)


def _iter_non_root(root) -> Iterator:
    stack: list = list(root.children)
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _message_text(message: dict[str, Any] | None) -> str:
    """Human-readable text for one node's chat message (``None`` for root /
    empty-response leaves)."""
    if not message:
        return ""
    content = message.get("content")
    parts: list[str] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append("reason:" + reasoning)
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
    elif content is not None:
        parts.append(str(content))
    return " ".join(p for p in parts if p)


def _render_subtree(node, lines: list[str], *, depth: int, max_text_chars: int) -> None:
    indent = "    " * depth
    for child in node.children:
        text = _message_text(child.message)
        if len(text) > max_text_chars:
            text = text[:max_text_chars] + "..."
        text = text.replace("'", "\\'")
        fields = [f"[{child.role}]"]
        if child.role == "assistant" and child.turn is not None:
            fields.append(f"turn={child.turn_index}")
            fields.append(f"prompt_ids=({len(child.turn.prompt_ids)})")
            fields.append(f"response_ids=({len(child.turn.output_ids)})")
            fields.append(f"finish={child.turn.finish_reason}")
            fields.append(f"has_logprobs={bool(child.turn.output_log_probs)}")
        fields.append(f"text='{text}'")
        lines.append(f"{indent}└── " + " ".join(fields))
        _render_subtree(child, lines, depth=depth + 1, max_text_chars=max_text_chars)
