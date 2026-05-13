from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_SLIME_ROOT = Path(__file__).resolve().parents[1]
if str(_SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLIME_ROOT))

from examples.sandbox_env.sglang_openai_proxy import SGLangOpenAIProxy, _anthropic_events


def test_anthropic_stream_events_include_text_and_tool_use_blocks() -> None:
    payload = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [
            {"type": "text", "text": "inspect files"},
            {
                "type": "tool_use",
                "id": "toolu_test",
                "name": "Glob",
                "input": {"pattern": "**/*.java"},
            },
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 3},
    }

    events = _anthropic_events(payload)

    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "inspect files"}
    assert events[4][1]["content_block"]["name"] == "Glob"
    assert events[5][1]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"pattern": "**/*.java"}',
    }
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"


def _proxy() -> SGLangOpenAIProxy:
    return SGLangOpenAIProxy(
        args=None,
        rollout_state={},
        loop=asyncio.new_event_loop(),
        model_name="test-model",
    )


def _record_tool_call_generation(proxy: SGLangOpenAIProxy) -> None:
    proxy._record_turn(
        body={"model": "test-model"},
        messages=[{"role": "user", "content": "inspect the repo"}],
        tools=[],
        assistant_text="inspect files",
        payload={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "inspect files",
                        "tool_calls": [
                            {
                                "id": "call_known",
                                "type": "function",
                                "function": {"name": "Glob", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
        finish_reason="tool_calls",
        tool_call_count=1,
        generation_meta={"finish_reason": {"type": "stop"}},
    )


def test_loss_mask_matches_prior_tool_call_generation_by_id() -> None:
    proxy = _proxy()
    try:
        _record_tool_call_generation(proxy)

        proxy._record_turn(
            body={"model": "test-model"},
            messages=[
                {
                    "role": "assistant",
                    "content": "inspect files",
                    "tool_calls": [
                        {
                            "id": "call_known",
                            "type": "function",
                            "function": {"name": "Glob", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_known", "content": "README.md"},
            ],
            tools=[],
            assistant_text="done",
            payload={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
            finish_reason="stop",
            tool_call_count=0,
            generation_meta={"finish_reason": {"type": "stop"}},
        )

        assert proxy._training_messages[0]["step_loss_mask"] == 1
    finally:
        proxy.loop.close()


def test_loss_mask_does_not_match_tool_call_generation_by_text_only() -> None:
    proxy = _proxy()
    try:
        _record_tool_call_generation(proxy)

        proxy._record_turn(
            body={"model": "test-model"},
            messages=[
                {"role": "assistant", "content": "inspect files"},
                {"role": "tool", "tool_call_id": "call_known", "content": "README.md"},
            ],
            tools=[],
            assistant_text="done",
            payload={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
            finish_reason="stop",
            tool_call_count=0,
            generation_meta={"finish_reason": {"type": "stop"}},
        )

        assert proxy._training_messages[0]["step_loss_mask"] == 0
    finally:
        proxy.loop.close()
