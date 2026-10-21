"""Responses replay, parallel/custom tools, SSE framing and token provenance."""

import asyncio
import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_agent._fakes import FakeSGLangServer, ScriptedTokenizer
from tests.test_agent.test_adapters import _parse_sse

from slime.agent.adapters.responses import ResponsesAdapter, _translate_input, _translate_tools
from slime.utils.types import Sample

NUM_GPUS = 0


def test_codex_developer_messages_fit_single_system_model_templates():
    translated = _translate_input(
        [
            {"role": "developer", "content": "Use tools."},
            {"role": "developer", "content": "Sandbox policy."},
            {"role": "user", "content": "Fix it."},
            {"role": "developer", "content": "Additional constraint."},
        ]
    )
    assert translated[0] == {"role": "system", "content": "Use tools.\n\nSandbox policy."}
    assert all(message["role"] != "system" for message in translated[1:])
    assert "Additional constraint." in translated[-1]["content"]


@pytest.mark.parametrize("stream", [False, True])
def test_responses_tool_replay_keeps_original_tokens_and_parallel_results(stream):
    async def run():
        raw = (
            "Inspecting.<tool_call><function=read><parameter=path>a</parameter></function></tool_call>"
            "<tool_call><function=read><parameter=path>b</parameter></function></tool_call>"
        )
        tok = ScriptedTokenizer(prompts=[[1, 2], [1, 2, 701, 702, 3]], outputs={(701, 702): raw, (703,): "Fixed."})
        async with FakeSGLangServer([[(-0.1, 701), (-0.2, 702)], [(-0.3, 703)]]) as upstream:
            adapter = ResponsesAdapter(tokenizer=tok, sglang_url=upstream.url)
            adapter.open_session("sid")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            body = {
                "input": [{"role": "user", "content": "Fix it"}],
                "model": "actor",
                "store": False,
                "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}],
            }

            async def post(payload):
                response = await client.post(
                    "/v1/responses", headers={"Authorization": "Bearer sid"}, json={**payload, "stream": stream}
                )
                assert response.status == 200, await response.text()
                if not stream:
                    return await response.json()
                events = _parse_sse(await response.text())
                assert [p["sequence_number"] for _, p in events] == list(range(len(events)))
                assert events[0][0] == "response.created"
                assert events[-1][0] == "response.completed"
                assert any(kind == "response.output_text.delta" for kind, _ in events)
                return events[-1][1]["response"]

            try:
                first = await post(body)
                calls = [item for item in first["output"] if item["type"] == "function_call"]
                assert len(calls) == 2  # No truncation to the first tool call.
                replay = (
                    body["input"]
                    + first["output"]
                    + [
                        {"type": "function_call_output", "call_id": call["call_id"], "output": value}
                        for call, value in zip(reversed(calls), ["contents b", "contents a"], strict=True)
                    ]
                )
                second = await post({**body, "input": replay})
                assert second["output"][0]["content"][0]["text"] == "Fixed."
                samples = await adapter.finish_session("sid", base_sample=Sample(index=0, prompt=""), reward=1)
            finally:
                await client.close()
        assert tok.rendered[1][0][-2:] == [
            {"role": "tool", "content": "contents a"},
            {"role": "tool", "content": "contents b"},
        ]
        assert upstream.routing_keys == ["sid", "sid"]
        trained = []
        for sample in samples:
            assert sample.response_length == len(sample.loss_mask) == len(sample.rollout_log_probs)
            trained.extend(
                (token, lp)
                for token, lp, mask in zip(
                    sample.tokens[-sample.response_length :], sample.rollout_log_probs, sample.loss_mask, strict=True
                )
                if mask
            )
        assert trained == [(701, -0.1), (702, -0.2), (703, -0.3)]

    asyncio.run(run())


def test_custom_namespaced_tool_roundtrip():
    from slime.agent.adapters.responses import _output_items
    from slime.agent.parsing import ParsedModelOutput

    tools = [
        {
            "type": "namespace",
            "name": "edit",
            "tools": [{"type": "custom", "name": "patch", "format": {"type": "text"}}],
        }
    ]
    schema = _translate_tools(tools)
    assert schema[0]["function"]["name"] == "edit.patch"
    assert schema[0]["function"]["parameters"]["required"] == ["input"]
    parsed = ParsedModelOutput(
        text="", reasoning="", tool_uses=[{"name": "edit.patch", "input": {"input": "a\nb"}}], ill_formed=False
    )
    output = _output_items(parsed, tools)
    assert output[0]["type"] == "custom_tool_call"
    assert output[0]["namespace"] == "edit"
    history = _translate_input(
        output + [{"type": "custom_tool_call_output", "call_id": output[0]["call_id"], "output": "applied"}]
    )
    assert history[0]["tool_calls"][0]["function"] == {"name": "edit.patch", "arguments": {"input": "a\nb"}}
    assert history[1] == {"role": "tool", "content": "applied"}


@pytest.mark.parametrize(
    "item",
    [
        {"type": "item_reference", "id": "old"},
        {"type": "function_call_output", "call_id": "missing", "output": "wrong"},
    ],
)
def test_unsupported_history_fails_instead_of_silently_losing_context(item):
    with pytest.raises(web.HTTPBadRequest):
        _translate_input([item])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
