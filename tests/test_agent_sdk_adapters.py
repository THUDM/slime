import asyncio
import sys
from pathlib import Path

import httpx
import pytest
from aiohttp.test_utils import TestClient, TestServer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent.adapters import anthropic, openai
from slime.agent.trajectory import TurnRecord


NUM_GPUS = 0


agents = pytest.importorskip("agents")
anthropic_sdk = pytest.importorskip("anthropic")
openai_sdk = pytest.importorskip("openai")


class SDKTokenizer:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.rendered: list[tuple[list[dict], list[dict] | None]] = []

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True):
        self.rendered.append((list(messages), tools))
        return list(range(1, len(messages) + 2))

    def decode(self, ids, skip_special_tokens=False):
        return self.outputs[ids[0] - 1]


@pytest.mark.integration
def test_openai_agents_sdk_responses_runs_tool_loop_against_adapter(monkeypatch):
    async def run_case():
        openai._closed.clear()
        openai._inflight.clear()
        calls = []

        async def fake_generate(prompt_ids, session, body, app, **kwargs):
            calls.append({"prompt_ids": list(prompt_ids), "body": body})
            return TurnRecord(
                prompt_ids=list(prompt_ids),
                output_ids=[len(calls)],
                finish_reason="stop",
                output_log_probs=[-0.1 * len(calls)],
            )

        monkeypatch.setattr(openai, "_generate", fake_generate)
        tokenizer = SDKTokenizer(
            [
                "<tool_call><function=lookup><parameter=query>slime</parameter></function></tool_call>",
                "final after tool",
            ]
        )
        app, store = openai.start(tokenizer=tokenizer, sglang_url="http://unused")
        client = TestClient(TestServer(app))
        await client.start_server()
        base_url = str(client.make_url("/v1/"))
        http_client = httpx.AsyncClient(trust_env=False)
        oai = openai_sdk.AsyncOpenAI(
            api_key="sdk-openai",
            base_url=base_url,
            max_retries=0,
            http_client=http_client,
        )

        @agents.function_tool
        def lookup(query: str) -> str:
            return f"found {query}"

        agents.set_tracing_disabled(True)
        model = agents.OpenAIResponsesModel(model="actor", openai_client=oai)
        agent = agents.Agent(
            name="sdk-responses",
            instructions="Use lookup.",
            model=model,
            tools=[lookup],
            model_settings=agents.ModelSettings(max_tokens=4),
        )
        try:
            result = await agents.Runner.run(agent, "find slime")
        finally:
            await client.close()
            await oai.close()

        segments = openai.pop_session_split(store, "sdk-openai")
        assert result.final_output == "final after tool"
        assert len(calls) == 2
        assert calls[0]["body"]["max_output_tokens"] == 4
        assert calls[0]["body"]["tools"][0]["name"] == "lookup"
        assert calls[1]["body"]["input"][-1] == {
            "call_id": calls[1]["body"]["input"][-2]["call_id"],
            "output": "found slime",
            "type": "function_call_output",
        }
        assert tokenizer.rendered[0][0] == [
            {"role": "system", "content": "Use lookup."},
            {"role": "user", "content": "find slime"},
        ]
        assert tokenizer.rendered[1][0][-1] == {
            "role": "tool",
            "content": "found slime",
            "tool_call_id": calls[1]["body"]["input"][-2]["call_id"],
        }
        assert segments[0].metadata["segment_kind"] == "final"
        assert segments[0].response_ids[-1] == 2
        assert segments[0].loss_mask[-1] == 1

    asyncio.run(run_case())


@pytest.mark.integration
def test_openai_agents_sdk_chat_completions_runs_against_adapter(monkeypatch):
    async def run_case():
        openai._closed.clear()
        openai._inflight.clear()
        calls = []

        async def fake_generate(prompt_ids, session, body, app, **kwargs):
            calls.append({"prompt_ids": list(prompt_ids), "body": body})
            return TurnRecord(
                prompt_ids=list(prompt_ids), output_ids=[1], finish_reason="stop", output_log_probs=[-0.2]
            )

        monkeypatch.setattr(openai, "_generate", fake_generate)
        tokenizer = SDKTokenizer(["chat final"])
        app, store = openai.start(tokenizer=tokenizer, sglang_url="http://unused")
        client = TestClient(TestServer(app))
        await client.start_server()
        base_url = str(client.make_url("/v1/"))
        http_client = httpx.AsyncClient(trust_env=False)
        oai = openai_sdk.AsyncOpenAI(
            api_key="sdk-openai-chat",
            base_url=base_url,
            max_retries=0,
            http_client=http_client,
        )

        agents.set_tracing_disabled(True)
        model = agents.OpenAIChatCompletionsModel(model="actor", openai_client=oai)
        agent = agents.Agent(
            name="sdk-chat",
            instructions="Be short.",
            model=model,
            model_settings=agents.ModelSettings(max_tokens=5),
        )
        try:
            result = await agents.Runner.run(agent, "say hi")
        finally:
            await client.close()
            await oai.close()

        segments = openai.pop_session_split(store, "sdk-openai-chat")
        assert result.final_output == "chat final"
        assert calls[0]["body"]["max_tokens"] == 5
        assert calls[0]["body"]["messages"] == [
            {"content": "Be short.", "role": "system"},
            {"role": "user", "content": "say hi"},
        ]
        assert segments[0].prompt_ids == [1, 2, 3]
        assert segments[0].response_ids == [1]
        assert segments[0].loss_mask == [1]

    asyncio.run(run_case())


@pytest.mark.integration
def test_openai_sdk_chat_completion_streaming_runs_against_adapter(monkeypatch):
    async def run_case():
        openai._closed.clear()
        openai._inflight.clear()
        calls = []

        async def fake_generate(prompt_ids, session, body, app, **kwargs):
            calls.append({"prompt_ids": list(prompt_ids), "body": body})
            return TurnRecord(
                prompt_ids=list(prompt_ids), output_ids=[1], finish_reason="stop", output_log_probs=[-0.25]
            )

        monkeypatch.setattr(openai, "_generate", fake_generate)
        tokenizer = SDKTokenizer(
            ["streamed via sdk <tool_call><function=lookup><parameter=query>slime</parameter></function></tool_call>"]
        )
        app, store = openai.start(tokenizer=tokenizer, sglang_url="http://unused")
        client = TestClient(TestServer(app))
        await client.start_server()
        base_url = str(client.make_url("/v1/"))
        http_client = httpx.AsyncClient(trust_env=False)
        oai = openai_sdk.AsyncOpenAI(
            api_key="sdk-openai-chat-stream",
            base_url=base_url,
            max_retries=0,
            http_client=http_client,
        )

        try:
            stream = await oai.chat.completions.create(
                model="actor",
                messages=[{"role": "user", "content": "call lookup"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "search",
                            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                        },
                    }
                ],
                stream=True,
            )
            text_parts = []
            tool_names = []
            tool_arguments = []
            finish_reasons = []
            usages = []
            async for chunk in stream:
                choice = chunk.choices[0]
                if choice.delta.content:
                    text_parts.append(choice.delta.content)
                if choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        tool_names.append(tool_call.function.name)
                        tool_arguments.append(tool_call.function.arguments)
                if choice.finish_reason:
                    finish_reasons.append(choice.finish_reason)
                if chunk.usage:
                    usages.append(chunk.usage)
        finally:
            await client.close()
            await oai.close()

        segments = openai.pop_session_split(store, "sdk-openai-chat-stream")
        assert "".join(text_parts) == "streamed via sdk"
        assert tool_names == ["lookup"]
        assert tool_arguments == ['{"query": "slime"}']
        assert finish_reasons == ["tool_calls"]
        assert usages[-1].prompt_tokens == 2
        assert usages[-1].completion_tokens == 1
        assert calls[0]["body"]["stream"] is True
        assert segments[0].response_ids == [1]

    asyncio.run(run_case())


@pytest.mark.integration
def test_openai_sdk_responses_streaming_runs_against_adapter(monkeypatch):
    async def run_case():
        openai._closed.clear()
        openai._inflight.clear()
        calls = []

        async def fake_generate(prompt_ids, session, body, app, **kwargs):
            calls.append({"prompt_ids": list(prompt_ids), "body": body})
            return TurnRecord(
                prompt_ids=list(prompt_ids), output_ids=[1], finish_reason="stop", output_log_probs=[-0.35]
            )

        monkeypatch.setattr(openai, "_generate", fake_generate)
        tokenizer = SDKTokenizer(["response stream via sdk"])
        app, store = openai.start(tokenizer=tokenizer, sglang_url="http://unused")
        client = TestClient(TestServer(app))
        await client.start_server()
        base_url = str(client.make_url("/v1/"))
        http_client = httpx.AsyncClient(trust_env=False)
        oai = openai_sdk.AsyncOpenAI(
            api_key="sdk-openai-responses-stream",
            base_url=base_url,
            max_retries=0,
            http_client=http_client,
        )

        try:
            stream = await oai.responses.create(
                model="actor",
                instructions="Be brief.",
                input="say hi",
                stream=True,
            )
            event_types = []
            deltas = []
            completed_response = None
            async for event in stream:
                event_types.append(event.type)
                if event.type == "response.output_text.delta":
                    deltas.append(event.delta)
                if event.type == "response.completed":
                    completed_response = event.response
        finally:
            await client.close()
            await oai.close()

        segments = openai.pop_session_split(store, "sdk-openai-responses-stream")
        assert event_types == ["response.created", "response.output_text.delta", "response.completed"]
        assert "".join(deltas) == "response stream via sdk"
        assert completed_response.status == "completed"
        assert completed_response.usage.input_tokens == 3
        assert completed_response.usage.output_tokens == 1
        assert calls[0]["body"]["stream"] is True
        assert segments[0].response_ids == [1]

    asyncio.run(run_case())


@pytest.mark.integration
def test_anthropic_sdk_non_streaming_messages_runs_against_adapter(monkeypatch):
    async def run_case():
        anthropic._closed.clear()
        anthropic._inflight.clear()
        calls = []

        async def fake_generate(prompt_ids, session, body, app, **kwargs):
            calls.append({"prompt_ids": list(prompt_ids), "body": body})
            return TurnRecord(
                prompt_ids=list(prompt_ids), output_ids=[1], finish_reason="stop", output_log_probs=[-0.28]
            )

        monkeypatch.setattr(anthropic, "_generate", fake_generate)
        tokenizer = SDKTokenizer(["anthropic json"])
        app, store = anthropic.start(tokenizer=tokenizer, sglang_url="http://unused")
        client = TestClient(TestServer(app))
        await client.start_server()
        base_url = str(client.make_url("/"))
        http_client = httpx.AsyncClient(trust_env=False)
        anth = anthropic_sdk.AsyncAnthropic(
            api_key="sdk-anthropic-json",
            base_url=base_url,
            max_retries=0,
            http_client=http_client,
        )

        try:
            message = await anth.messages.create(
                model="actor",
                max_tokens=6,
                system="Be direct.",
                messages=[{"role": "user", "content": "say hi"}],
            )
        finally:
            await client.close()
            await anth.close()

        segments = anthropic.pop_session_split(store, "sdk-anthropic-json")
        assert message.type == "message"
        assert message.content[0].type == "text"
        assert message.content[0].text == "anthropic json"
        assert message.stop_reason == "end_turn"
        assert message.usage.input_tokens == 3
        assert message.usage.output_tokens == 1
        assert calls[0]["body"]["max_tokens"] == 6
        assert segments[0].response_ids == [1]

    asyncio.run(run_case())


@pytest.mark.integration
def test_anthropic_sdk_streaming_messages_runs_against_adapter(monkeypatch):
    async def run_case():
        anthropic._closed.clear()
        anthropic._inflight.clear()
        calls = []

        async def fake_generate(prompt_ids, session, body, app, **kwargs):
            calls.append({"prompt_ids": list(prompt_ids), "body": body})
            return TurnRecord(
                prompt_ids=list(prompt_ids), output_ids=[1], finish_reason="stop", output_log_probs=[-0.3]
            )

        monkeypatch.setattr(anthropic, "_generate", fake_generate)
        tokenizer = SDKTokenizer(["anthropic final"])
        app, store = anthropic.start(tokenizer=tokenizer, sglang_url="http://unused")
        client = TestClient(TestServer(app))
        await client.start_server()
        base_url = str(client.make_url("/"))
        http_client = httpx.AsyncClient(trust_env=False)
        anth = anthropic_sdk.AsyncAnthropic(
            api_key="sdk-anthropic",
            base_url=base_url,
            max_retries=0,
            http_client=http_client,
        )

        try:
            stream = await anth.messages.create(
                model="actor",
                max_tokens=6,
                system="Be direct.",
                messages=[{"role": "user", "content": "say hi"}],
                stream=True,
            )
            text_parts = []
            event_types = []
            async for event in stream:
                event_types.append(event.type)
                if event.type == "content_block_delta" and getattr(event.delta, "text", None):
                    text_parts.append(event.delta.text)
        finally:
            await client.close()
            await anth.close()

        segments = anthropic.pop_session_split(store, "sdk-anthropic")
        assert "".join(text_parts) == "anthropic final"
        assert event_types == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert calls[0]["body"]["max_tokens"] == 6
        assert tokenizer.rendered[0][0] == [
            {"role": "system", "content": "Be direct."},
            {"role": "user", "content": "say hi"},
        ]
        assert segments[0].prompt_ids == [1, 2, 3]
        assert segments[0].response_ids == [1]
        assert segments[0].loss_mask == [1]

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
