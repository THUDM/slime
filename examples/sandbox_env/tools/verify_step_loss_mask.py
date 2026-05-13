#!/usr/bin/env python3
"""Verify step_loss_mask consistency in a sample_artifacts.json file.

For each assistant message in `final_messages`, classifies whether the message
was produced by the proxy itself (matched against `trajectory` / `turn_responses`)
and checks that `step_loss_mask` matches the expected value:
  proxy-produced, complete output -> step_loss_mask = 1
  proxy-produced, length-truncated -> step_loss_mask = 0
  proxy-produced, max-turn stop  -> step_loss_mask = 0
  harness-injected                -> step_loss_mask = 0

Matching uses the same normalization as `sglang_openai_proxy._normalize_assistant_for_match`:
strip <tool_call>...</tool_call> blocks, then strip whitespace.

Usage:
    python verify_step_loss_mask.py <sample_artifacts.json> [<label>]

Exit status: 0 on full match, 1 on any mismatch.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text") or ""))
                elif "content" in item:
                    parts.append(stringify_content(item.get("content")))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def normalize_for_match(content: Any) -> str:
    return TOOL_CALL_RE.sub("", stringify_content(content)).strip()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_step_loss_mask.py <sample_artifacts.json> [<label>]", file=sys.stderr)
        return 2
    artifact_path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else artifact_path

    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_messages = data.get("final_messages")
    if not isinstance(final_messages, list):
        final_messages = []
    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list):
        trajectory = []
    turn_responses = data.get("turn_responses")
    if not isinstance(turn_responses, list):
        turn_responses = []

    expected_turns: list[tuple[str, int, str]] = []
    for turn in trajectory:
        if not isinstance(turn, dict):
            continue
        text = turn.get("raw_model_response")
        if text is None:
            continue
        raw_finish = str(turn.get("raw_generation_finish_reason") or "")
        expected_mask = 0 if raw_finish in {"length", "max_turns"} else 1
        expected_turns.append((normalize_for_match(text), expected_mask, raw_finish))
    if not expected_turns:
        expected_turns = [(normalize_for_match(t), 1, "") for t in turn_responses]

    print(f"\n[{label}]")
    print(f"  final_messages: {len(final_messages)} entries")
    print(f"  turn_responses: {len(turn_responses)} entries (proxy-generated)")
    print(f"  trajectory: {len(trajectory)} entries")
    if data.get("final_messages") is None:
        print("  NOTE: final_messages is null; treating it as an empty message list.")
    if data.get("turn_responses") is None:
        print("  NOTE: turn_responses is null; treating it as an empty proxy response list.")

    # Skip-ahead cursor matching, mirroring sglang_openai_proxy: each assistant
    # in final_messages is matched against proxy turns by content equality
    # (after normalisation), scanning forward from the cursor so hidden proxy
    # calls such as title-generation requests that never appear in the visible
    # history are skipped past, not failed on.
    cursor = 0
    mismatches: list[tuple[int, int, int]] = []
    proxy_supervised_count = 0
    proxy_unsupervised_count = 0
    injected_count = 0

    for i, msg in enumerate(final_messages):
        if msg.get("role") != "assistant":
            continue
        norm = normalize_for_match(msg.get("content"))
        mark = msg.get("step_loss_mask", 1)

        found = next(
            (j for j in range(cursor, len(expected_turns)) if expected_turns[j][0] == norm),
            -1,
        )
        is_proxy = found >= 0
        if is_proxy:
            expected = expected_turns[found][1]
            cursor = found + 1
            if expected:
                proxy_supervised_count += 1
            else:
                proxy_unsupervised_count += 1
        else:
            expected = 0
            injected_count += 1

        flag = "OK  " if mark == expected else "FAIL"
        preview = (norm[:60] + "…") if len(norm) > 60 else norm
        print(f"    [{i}] mark={mark} expected={expected} {flag}  {preview!r}")
        if mark != expected:
            mismatches.append((i, mark, expected))

    print(
        "  proxy-produced supervised: "
        f"{proxy_supervised_count}, proxy-produced unsupervised: "
        f"{proxy_unsupervised_count}, harness-injected: {injected_count}"
    )

    if cursor < len(expected_turns):
        # Some proxy generations weren't matched in final_messages — could mean
        # the agent dropped them from history. Not necessarily an error, but worth
        # surfacing.
        print(
            f"  WARNING: {len(expected_turns) - cursor} proxy-generated text(s) "
            f"did not appear in final_messages (agent likely truncated history)."
        )

    if mismatches:
        print(f"  RESULT: FAIL ({len(mismatches)} mismatches)")
        return 1
    print("  RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
