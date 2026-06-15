import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent.adapters import anthropic, openai
from slime.agent.adapters.common import SGLANG_URL_KEY
from slime.agent.trajectory import TurnRecord


NUM_GPUS = 0


class ToyTokenizer:
    def __init__(self, outputs: dict[tuple[int, ...], str] | None = None) -> None:
        self.outputs = outputs or {}
        self.rendered: list[tuple[list[dict], list[dict] | None]] = []

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True):
        self.rendered.append((list(messages), tools))
        return list(range(1, len(messages) + 2))

    def decode(self, ids, skip_special_tokens=False):
        return self.outputs.get(tuple(ids), "")


class ScriptedTokenizer(ToyTokenizer):
    def __init__(self, prompts: list[list[int]], outputs: dict[tuple[int, ...], str]) -> None:
        super().__init__(outputs)
        self.prompts = [list(prompt) for prompt in prompts]

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True):
        self.rendered.append((list(messages), tools))
        assert self.prompts, "unexpected chat-template render"
        return self.prompts.pop(0)


class FakeSGLang:
    def __init__(self, turns: list[list[tuple[float, int]]]) -> None:
        self.turns = [list(turn) for turn in turns]
        self.requests: list[dict] = []
        self.routing_keys: list[str | None] = []

    async def handle_generate(self, request):
        self.routing_keys.append(request.headers.get("X-SMG-Routing-Key"))
        self.requests.append(await request.json())
        assert self.turns, "unexpected /generate call"
        output_token_logprobs = [[logprob, token_id] for logprob, token_id in self.turns.pop(0)]
        return web.json_response(
            {
                "meta_info": {
                    "output_token_logprobs": output_token_logprobs,
                    "finish_reason": {"type": "stop"},
                }
            }
        )


class FakeRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def _parse_sse(raw: str) -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = []
    event_name = "message"
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return
        data = "\n".join(data_lines)
        payload: object
        if data == "[DONE]":
            payload = data
        else:
            payload = json.loads(data)
        events.append((event_name, payload))
        event_name = "message"
        data_lines = []

    for line in raw.splitlines():
        if not line:
            flush()
        elif line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    flush()
    return events


@pytest.mark.unit
def test_session_id_comes_from_protocol_fields_not_custom_header():
    assert (
        openai._request_session_id(
            FakeRequest({"X-Slime-Session-Id": "custom"}),
            {"metadata": {"session_id": "meta-session"}, "user": "body-user"},
        )
        == "meta-session"
    )
    assert (
        openai._request_session_id(FakeRequest({"X-Slime-Session-Id": "custom"}), {"user": "body-user"}) == "body-user"
    )
    assert (
        anthropic._request_session_id(FakeRequest({"X-Slime-Session-Id": "custom", "X-Api-Key": "anthropic-key"}))
        == "anthropic-key"
    )
    assert (
        anthropic._request_session_id(
            FakeRequest({"Authorization": "Bearer bearer-session", "X-Api-Key": "anthropic-key"})
        )
        == "bearer-session"
    )


@pytest.mark.unit
def test_anthropic_translation_keeps_tool_results_and_tool_schema():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "name": "lookup", "input": {"q": "slime"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "result"}]},
    ]

    translated = anthropic._translate_anthropic(messages, system="sys")
    tools = anthropic._anthropic_tools_to_chat_tools(
        [{"name": "lookup", "description": "search", "input_schema": {"type": "object"}}]
    )

    assert translated == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "plan",
            "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "slime"}}}],
        },
        {"role": "tool", "content": "result"},
    ]
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "search",
                "parameters": {"type": "object"},
            },
        }
    ]


@pytest.mark.unit
def test_openai_translation_and_responses_input_shapes():
    chat_messages = openai._translate_chat_messages(
        [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"q": "slime"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "found"},
        ]
    )
    response_messages = openai._responses_input_to_messages(
        [
            {"role": "user", "content": [{"type": "input_text", "text": "question"}]},
            {"type": "function_call_output", "call_id": "call_1", "output": "answer"},
        ],
        instructions="be brief",
    )

    assert chat_messages == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "slime"}'},
                }
            ],
        },
        {"role": "tool", "content": "found", "tool_call_id": "call_1"},
    ]
    assert response_messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "input_text", "text": "question"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "answer"},
    ]


@pytest.mark.unit
def test_openai_chat_completion_endpoint_records_token_segments(monkeypatch):
    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[101], finish_reason="stop", output_log_probs=[-0.1])

    async def run_case():
        monkeypatch.setattr(openai, "_generate", fake_generate)
        tokenizer = ToyTokenizer({(101,): "hello"})
        adapter = openai.OpenAIAdapter(tokenizer=tokenizer, sglang_url="http://unused")
        adapter.open_session("sid-chat", sampling_defaults={"max_new_tokens": 8})
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sid-chat"},
                json={
                    "model": "actor",
                    "messages": [{"role": "user", "content": "hello?"}],
                    "max_tokens": 4,
                },
            )
            data = await resp.json()
        finally:
            await client.close()

        segments = await adapter.finish_session("sid-chat")
        assert resp.status == 200
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
        assert data["usage"] == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        assert segments[0].prompt_ids == [1, 2]
        assert segments[0].response_ids == [101]
        assert segments[0].loss_mask == [1]

    asyncio.run(run_case())


@pytest.mark.unit
def test_openai_chat_completion_streaming_returns_sse_chunks_and_records_segments(monkeypatch):
    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[401], finish_reason="stop", output_log_probs=[-0.4])

    async def run_case():
        monkeypatch.setattr(openai, "_generate", fake_generate)
        tokenizer = ToyTokenizer({(401,): "streamed text"})
        adapter = openai.OpenAIAdapter(tokenizer=tokenizer, sglang_url="http://unused")
        adapter.open_session("sid-chat-stream", sampling_defaults={"max_new_tokens": 8})
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sid-chat-stream"},
                json={
                    "model": "actor",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello?"}],
                },
            )
            raw = await resp.text()
        finally:
            await client.close()

        events = _parse_sse(raw)
        chunks = [payload for _, payload in events if isinstance(payload, dict)]
        segments = await adapter.finish_session("sid-chat-stream")
        assert resp.status == 200
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert any(c["choices"][0]["delta"] == {"content": "streamed text"} for c in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["usage"] == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        assert events[-1] == ("message", "[DONE]")
        assert segments[0].prompt_ids == [1, 2]
        assert segments[0].response_ids == [401]
        assert segments[0].rollout_log_probs == [-0.4]

    asyncio.run(run_case())


@pytest.mark.unit
def test_openai_chat_completion_streaming_returns_tool_call_delta(monkeypatch):
    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(
            prompt_ids=list(prompt_ids), output_ids=[451], finish_reason="stop", output_log_probs=[-0.45]
        )

    async def run_case():
        monkeypatch.setattr(openai, "_generate", fake_generate)
        raw = "use it <tool_call><function=lookup><parameter=query>slime</parameter></function></tool_call>"
        tokenizer = ToyTokenizer({(451,): raw})
        adapter = openai.OpenAIAdapter(tokenizer=tokenizer, sglang_url="http://unused")
        adapter.open_session("sid-chat-tool-stream", sampling_defaults={"max_new_tokens": 8})
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sid-chat-tool-stream"},
                json={
                    "model": "actor",
                    "stream": True,
                    "messages": [{"role": "user", "content": "call lookup"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "description": "search",
                                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                            },
                        }
                    ],
                },
            )
            raw_sse = await resp.text()
        finally:
            await client.close()

        chunks = [payload for _, payload in _parse_sse(raw_sse) if isinstance(payload, dict)]
        tool_delta = next(c["choices"][0]["delta"] for c in chunks if "tool_calls" in c["choices"][0]["delta"])
        segments = await adapter.finish_session("sid-chat-tool-stream")
        assert resp.status == 200
        assert any(c["choices"][0]["delta"] == {"content": "use it"} for c in chunks)
        assert tool_delta["tool_calls"][0]["index"] == 0
        assert tool_delta["tool_calls"][0]["function"]["name"] == "lookup"
        assert tool_delta["tool_calls"][0]["function"]["arguments"] == '{"query": "slime"}'
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert segments[0].response_ids == [451]

    asyncio.run(run_case())


@pytest.mark.unit
def test_openai_responses_endpoint_returns_function_calls(monkeypatch):
    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[301], finish_reason="stop", output_log_probs=[-0.3])

    async def run_case():
        monkeypatch.setattr(openai, "_generate", fake_generate)
        raw = "look <tool_call><function=lookup><parameter=query>slime</parameter></function></tool_call>"
        tokenizer = ToyTokenizer({(301,): raw})
        adapter = openai.OpenAIAdapter(tokenizer=tokenizer, sglang_url="http://unused")
        adapter.open_session("sid-responses", sampling_defaults={"max_new_tokens": 8})
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            resp = await client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer sid-responses"},
                json={
                    "model": "actor",
                    "input": "find it",
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup",
                            "description": "search",
                            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                        }
                    ],
                },
            )
            data = await resp.json()
        finally:
            await client.close()

        output_types = [item["type"] for item in data["output"]]
        function_call = next(item for item in data["output"] if item["type"] == "function_call")
        segments = await adapter.finish_session("sid-responses")
        assert resp.status == 200
        assert data["object"] == "response"
        assert output_types == ["message", "function_call"]
        assert data["output"][0]["content"][0]["text"] == "look"
        assert function_call["name"] == "lookup"
        assert function_call["arguments"] == '{"query": "slime"}'
        assert segments[0].response_ids == [301]

    asyncio.run(run_case())


@pytest.mark.unit
def test_openai_responses_streaming_preserves_function_call_output(monkeypatch):
    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(
            prompt_ids=list(prompt_ids), output_ids=[551], finish_reason="stop", output_log_probs=[-0.55]
        )

    async def run_case():
        monkeypatch.setattr(openai, "_generate", fake_generate)
        raw = "<tool_call><function=lookup><parameter=query>slime</parameter></function></tool_call>"
        tokenizer = ToyTokenizer({(551,): raw})
        adapter = openai.OpenAIAdapter(tokenizer=tokenizer, sglang_url="http://unused")
        adapter.open_session("sid-responses-tool-stream", sampling_defaults={"max_new_tokens": 8})
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            resp = await client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer sid-responses-tool-stream"},
                json={
                    "model": "actor",
                    "stream": True,
                    "input": "call lookup",
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                        }
                    ],
                },
            )
            raw_sse = await resp.text()
        finally:
            await client.close()

        events = _parse_sse(raw_sse)
        created = next(payload for name, payload in events if name == "response.created")
        completed = next(payload for name, payload in events if name == "response.completed")
        completed_call = next(item for item in completed["response"]["output"] if item["type"] == "function_call")
        segments = await adapter.finish_session("sid-responses-tool-stream")
        assert resp.status == 200
        assert created["type"] == "response.created"
        assert [item["type"] for item in created["response"]["output"]] == ["function_call"]
        assert completed_call["name"] == "lookup"
        assert completed_call["arguments"] == '{"query": "slime"}'
        assert segments[0].response_ids == [551]

    asyncio.run(run_case())


@pytest.mark.unit
def test_openai_responses_streaming_returns_sse_events_and_records_segments(monkeypatch):
    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[501], finish_reason="stop", output_log_probs=[-0.5])

    async def run_case():
        monkeypatch.setattr(openai, "_generate", fake_generate)
        tokenizer = ToyTokenizer({(501,): "response text"})
        adapter = openai.OpenAIAdapter(tokenizer=tokenizer, sglang_url="http://unused")
        adapter.open_session("sid-responses-stream", sampling_defaults={"max_new_tokens": 8})
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            resp = await client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer sid-responses-stream"},
                json={
                    "model": "actor",
                    "stream": True,
                    "instructions": "be brief",
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello?"}]}],
                },
            )
            raw = await resp.text()
        finally:
            await client.close()

        events = _parse_sse(raw)
        event_names = [name for name, _ in events]
        text_delta = next(payload for name, payload in events if name == "response.output_text.delta")
        completed = next(payload for name, payload in events if name == "response.completed")
        segments = await adapter.finish_session("sid-responses-stream")
        assert resp.status == 200
        assert event_names == ["response.created", "response.output_text.delta", "response.completed"]
        assert text_delta == {"type": "response.output_text.delta", "delta": "response text"}
        assert completed["response"]["status"] == "completed"
        assert completed["response"]["usage"] == {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
        assert segments[0].prompt_ids == [1, 2, 3]
        assert segments[0].response_ids == [501]

    asyncio.run(run_case())


@pytest.mark.unit
def test_anthropic_messages_endpoint_returns_non_stream_json_and_records_segments(monkeypatch):
    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(
            prompt_ids=list(prompt_ids), output_ids=[581], finish_reason="stop", output_log_probs=[-0.58]
        )

    async def run_case():
        monkeypatch.setattr(anthropic, "_generate", fake_generate)
        tokenizer = ToyTokenizer({(581,): "plain response"})
        adapter = anthropic.AnthropicAdapter(tokenizer=tokenizer, sglang_url="http://unused")
        adapter.open_session("sid-anthropic-json", sampling_defaults={"max_new_tokens": 8})
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            resp = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer sid-anthropic-json"},
                json={
                    "model": "actor",
                    "system": "be useful",
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "solve"}]}],
                },
            )
            data = await resp.json()
        finally:
            await client.close()

        segments = await adapter.finish_session("sid-anthropic-json")
        assert resp.status == 200
        assert data["type"] == "message"
        assert data["model"] == "actor"
        assert data["content"] == [{"type": "text", "text": "plain response"}]
        assert data["stop_reason"] == "end_turn"
        assert data["usage"] == {"input_tokens": 3, "output_tokens": 1}
        assert segments[0].prompt_ids == [1, 2, 3]
        assert segments[0].response_ids == [581]

    asyncio.run(run_case())


@pytest.mark.unit
def test_anthropic_messages_endpoint_streams_blocks_and_records_segments(monkeypatch):
    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[601], finish_reason="stop", output_log_probs=[-0.6])

    async def run_case():
        monkeypatch.setattr(anthropic, "_generate", fake_generate)
        raw_output = (
            "delegate <tool_call><function=Task><parameter=description>inspect</parameter></function></tool_call>"
        )
        tokenizer = ToyTokenizer({(601,): raw_output})
        adapter = anthropic.AnthropicAdapter(tokenizer=tokenizer, sglang_url="http://unused")
        adapter.open_session("sid-anthropic-stream", sampling_defaults={"max_new_tokens": 8})
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            resp = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer sid-anthropic-stream"},
                json={
                    "model": "actor",
                    "system": "be useful",
                    "stream": True,
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "solve"}]}],
                    "tools": [
                        {
                            "name": "Task",
                            "description": "spawn subagent",
                            "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}},
                        }
                    ],
                },
            )
            raw = await resp.text()
        finally:
            await client.close()

        events = _parse_sse(raw)
        names = [name for name, _ in events]
        starts = [payload for name, payload in events if name == "content_block_start"]
        deltas = [payload for name, payload in events if name == "content_block_delta"]
        message_delta = next(payload for name, payload in events if name == "message_delta")
        segments = await adapter.finish_session("sid-anthropic-stream")
        assert resp.status == 200
        assert names[0] == "message_start"
        assert names[-1] == "message_stop"
        assert any(s["content_block"]["type"] == "text" for s in starts)
        assert any(s["content_block"]["type"] == "tool_use" and s["content_block"]["name"] == "Task" for s in starts)
        assert any(d["delta"].get("text") == "delegate" for d in deltas)
        assert any(json.loads(d["delta"].get("partial_json", "{}")) == {"description": "inspect"} for d in deltas)
        assert message_delta["delta"]["stop_reason"] == "tool_use"
        assert message_delta["usage"] == {"input_tokens": 3, "output_tokens": 1}
        assert segments[0].metadata["segment_kind"] == "final"
        assert segments[0].prompt_ids == [1, 2, 3]
        assert segments[0].response_ids == [601]
        assert segments[0].loss_mask == [1]

    asyncio.run(run_case())


@pytest.mark.unit
def test_openai_responses_multiturn_uses_sglang_tokens_for_training_segment():
    async def run_case():
        upstream = FakeSGLang(
            [
                [(-0.20, 20), (-0.21, 21)],
                [(-0.40, 40)],
            ]
        )
        upstream_app = web.Application()
        upstream_app.router.add_post("/generate", upstream.handle_generate)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        tool_raw = "<tool_call><function=lookup><parameter=query>slime</parameter></function></tool_call>"
        tokenizer = ScriptedTokenizer(
            prompts=[
                [10, 11],
                [10, 11, 20, 21, 30, 31],
            ],
            outputs={
                (20, 21): tool_raw,
                (40,): "done",
            },
        )
        adapter = openai.OpenAIAdapter(tokenizer=tokenizer, sglang_url=str(upstream_server.make_url("")).rstrip("/"))
        adapter.open_session("sid-openai-token", sampling_defaults={"max_new_tokens": 99})
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            first = await client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer sid-openai-token"},
                json={
                    "model": "actor",
                    "input": "find slime",
                    "max_output_tokens": 5,
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                        }
                    ],
                },
            )
            first_data = await first.json()
            function_call = next(item for item in first_data["output"] if item["type"] == "function_call")

            second = await client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer sid-openai-token"},
                json={
                    "model": "actor",
                    "input": [
                        {"role": "user", "content": "find slime"},
                        function_call,
                        {
                            "type": "function_call_output",
                            "call_id": function_call["call_id"],
                            "output": "found slime",
                        },
                    ],
                    "max_output_tokens": 7,
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                        }
                    ],
                },
            )
            second_data = await second.json()
        finally:
            await client.close()
            await upstream_server.close()

        segments = await adapter.finish_session("sid-openai-token")
        assert first.status == 200
        assert second.status == 200
        assert function_call["name"] == "lookup"
        assert function_call["arguments"] == '{"query": "slime"}'
        assert second_data["output"][0]["content"][0]["text"] == "done"
        assert [req["input_ids"] for req in upstream.requests] == [[10, 11], [10, 11, 20, 21, 30, 31]]
        assert upstream.routing_keys == ["sid-openai-token", "sid-openai-token"]
        assert upstream.requests[0]["sampling_params"]["max_new_tokens"] == 5
        assert upstream.requests[1]["sampling_params"]["max_new_tokens"] == 7
        assert segments[0].prompt_ids == [10, 11]
        assert segments[0].response_ids == [20, 21, 30, 31, 40]
        assert segments[0].loss_mask == [1, 1, 0, 0, 1]
        assert segments[0].rollout_log_probs == [-0.20, -0.21, 0.0, 0.0, -0.40]

    asyncio.run(run_case())


@pytest.mark.unit
def test_anthropic_messages_multiturn_uses_sglang_tokens_for_training_segment():
    async def run_case():
        upstream = FakeSGLang(
            [
                [(-1.20, 120), (-1.21, 121)],
                [(-1.40, 140), (-1.41, 141)],
            ]
        )
        upstream_app = web.Application()
        upstream_app.router.add_post("/generate", upstream.handle_generate)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        tool_raw = "<tool_call><function=lookup><parameter=query>slime</parameter></function></tool_call>"
        tokenizer = ScriptedTokenizer(
            prompts=[
                [110, 111],
                [110, 111, 120, 121, 130],
            ],
            outputs={
                (120, 121): tool_raw,
                (140, 141): "anthropic done",
            },
        )
        adapter = anthropic.AnthropicAdapter(
            tokenizer=tokenizer,
            sglang_url=str(upstream_server.make_url("")).rstrip("/"),
        )
        adapter.open_session("sid-anthropic-token", sampling_defaults={"max_new_tokens": 99})
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            first = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer sid-anthropic-token"},
                json={
                    "model": "actor",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "find slime"}]}],
                    "tools": [
                        {
                            "name": "lookup",
                            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                        }
                    ],
                },
            )
            first_data = await first.json()
            tool_use = next(block for block in first_data["content"] if block["type"] == "tool_use")

            second = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer sid-anthropic-token"},
                json={
                    "model": "actor",
                    "max_tokens": 7,
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "find slime"}]},
                        {"role": "assistant", "content": first_data["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use["id"],
                                    "content": "found slime",
                                }
                            ],
                        },
                    ],
                    "tools": [
                        {
                            "name": "lookup",
                            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                        }
                    ],
                },
            )
            second_data = await second.json()
        finally:
            await client.close()
            await upstream_server.close()

        segments = await adapter.finish_session("sid-anthropic-token")
        assert first.status == 200
        assert second.status == 200
        assert tool_use["name"] == "lookup"
        assert tool_use["input"] == {"query": "slime"}
        assert second_data["content"] == [{"type": "text", "text": "anthropic done"}]
        assert [req["input_ids"] for req in upstream.requests] == [[110, 111], [110, 111, 120, 121, 130]]
        assert upstream.routing_keys == ["sid-anthropic-token", "sid-anthropic-token"]
        assert upstream.requests[0]["sampling_params"]["max_new_tokens"] == 5
        assert upstream.requests[1]["sampling_params"]["max_new_tokens"] == 7
        assert segments[0].prompt_ids == [110, 111]
        assert segments[0].response_ids == [120, 121, 130, 140, 141]
        assert segments[0].loss_mask == [1, 1, 0, 1, 1]
        assert segments[0].rollout_log_probs == [-1.20, -1.21, 0.0, -1.40, -1.41]

    asyncio.run(run_case())


@pytest.mark.unit
def test_openai_generate_posts_input_ids_and_extracts_logprobs():
    async def run_case():
        captured = {}
        captured_headers = {}

        async def handle_generate(request):
            captured_headers.update(request.headers)
            captured.update(await request.json())
            return web.json_response(
                {
                    "meta_info": {
                        "output_token_logprobs": [[-0.7, 701], [-0.8, 702]],
                        "finish_reason": {"type": "stop"},
                    }
                }
            )

        upstream_app = web.Application()
        upstream_app.router.add_post("/generate", handle_generate)
        server = TestServer(upstream_app)
        await server.start_server()
        try:
            session = openai.Session(sampling_defaults={"max_new_tokens": 9})
            turn = await openai._generate(
                [11, 12],
                session,
                {"max_tokens": 3, "temperature": 0.25, "stop": ["</s>"]},
                {SGLANG_URL_KEY: str(server.make_url("")).rstrip("/")},
            )
        finally:
            await server.close()

        assert captured["input_ids"] == [11, 12]
        assert captured["return_logprob"] is True
        assert captured_headers.get("X-SMG-Routing-Key") is None
        assert captured["sampling_params"]["max_new_tokens"] == 3
        assert captured["sampling_params"]["temperature"] == 0.25
        assert captured["sampling_params"]["stop"] == ["</s>"]
        assert turn.prompt_ids == [11, 12]
        assert turn.output_ids == [701, 702]
        assert turn.output_log_probs == [-0.7, -0.8]

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
