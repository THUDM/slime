"""Original model tokens survive text-only agent history round trips."""

import asyncio
import copy
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_e2e_helpers import audit_token_records
from tests.test_agent._fakes import FakeSGLangServer

from slime.agent.adapters.anthropic import AnthropicAdapter
from slime.agent.adapters.common import PromptPrefix, Session, _render_token_ids, _session_prompt_ids
from slime.agent.adapters.responses import ResponsesAdapter
from slime.agent.trajectory import TurnRecord
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    special = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    backend.train_from_iterator(
        ["assistant user system tool\nRead the files.\n\nCheck both paths and preserve the sampled reasoning."],
        trainers.BpeTrainer(
            vocab_size=350, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=special
        ),
    )
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>", additional_special_tokens=special)
    tok.chat_template = (
        "{% if tools %}{{ '<|im_start|>system\\n' + (tools|tojson) + '<|im_end|>\\n' }}{% endif %}"
        "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' }}"
        "{% if m.role == 'assistant' %}{{ '<think>\\n' + (m.reasoning_content|default('')) + '\\n</think>\\n\\n' }}{% endif %}"
        "{{ m.content }}{% if m.tool_calls %}{{ m.tool_calls|tojson }}{% endif %}{{ '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n' }}{% endif %}"
    )
    return tok


def completed_prefix(tokenizer, session, messages, tools=None):
    prompt = _session_prompt_ids(messages, tokenizer, tools=tools, session=session)
    output = tokenizer.encode("Keep this exact reasoning.\n</think>\n\nReading. <|im_end|>", add_special_tokens=False)
    turn = TurnRecord(prompt, output, "stop", [-0.1] * len(output))
    history = messages + [{"role": "assistant", "content": "Reading."}]
    session.prompt_prefixes.append(PromptPrefix(copy.deepcopy(history), copy.deepcopy(tools), turn))
    return history, turn


def test_reasoning_and_noncanonical_whitespace_are_reused_before_sampling(tokenizer):
    session = Session()
    history, first = completed_prefix(tokenizer, session, [{"role": "user", "content": "Read the files."}])
    history.append({"role": "tool", "content": "file contents"})
    canonical = _render_token_ids(history, tokenizer, tools=None)
    next_prompt = _session_prompt_ids(history, tokenizer, tools=None, session=session)
    original = first.prompt_ids + first.output_ids
    assert canonical[: len(original)] != original
    assert next_prompt[: len(original)] == original
    assert (
        tokenizer.decode(next_prompt[len(original) :])
        == "\n<|im_start|>tool\nfile contents<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    second_history, second = completed_prefix(tokenizer, session, history)
    second_history.append({"role": "tool", "content": "more files"})
    third_prompt = _session_prompt_ids(second_history, tokenizer, tools=None, session=session)
    assert third_prompt[: len(second.prompt_ids + second.output_ids)] == second.prompt_ids + second.output_ids


@pytest.mark.parametrize("change", ["user", "assistant", "tools", "compaction", "unfinished", "no_boundary"])
def test_changed_or_unfinished_history_is_not_spliced(tokenizer, change):
    session = Session()
    tools = [{"type": "function", "function": {"name": "read"}}]
    history, turn = completed_prefix(tokenizer, session, [{"role": "user", "content": "Read the files."}], tools)
    history.append({"role": "tool", "content": "file contents"})
    if change in {"user", "assistant"}:
        history[0 if change == "user" else 1]["content"] = "Changed."
    elif change == "tools":
        tools = [{"type": "function", "function": {"name": "write"}}]
    elif change == "compaction":
        history = [{"role": "user", "content": "Summary: already read the files."}]
    else:
        from dataclasses import replace

        session.prompt_prefixes[-1].turn = (
            replace(turn, finish_reason="length")
            if change == "unfinished"
            else replace(turn, output_ids=turn.output_ids[:-1])
        )
    assert _session_prompt_ids(history, tokenizer, tools=tools, session=session) == _render_token_ids(
        history, tokenizer, tools=tools
    )


def test_sibling_tool_results_branch_from_the_original_prefix(tokenizer):
    session = Session()
    root, first = completed_prefix(tokenizer, session, [{"role": "user", "content": "Read the files."}])
    left = root + [{"role": "tool", "content": "left branch"}]
    completed_prefix(tokenizer, session, left)
    right = root + [{"role": "tool", "content": "right branch"}]
    right_prompt = _session_prompt_ids(right, tokenizer, tools=None, session=session)
    original = first.prompt_ids + first.output_ids
    assert right_prompt[: len(original)] == original
    assert "left branch" not in tokenizer.decode(right_prompt)
    assert "right branch" in tokenizer.decode(right_prompt)


def test_identical_visible_replies_with_different_reasoning_are_ambiguous(tokenizer):
    from dataclasses import replace

    session = Session()
    history, turn = completed_prefix(tokenizer, session, [{"role": "user", "content": "Read the files."}])
    other = replace(
        turn,
        output_ids=tokenizer.encode("Different reasoning.\n</think>\n\nReading.<|im_end|>", add_special_tokens=False),
    )
    session.prompt_prefixes.append(PromptPrefix(copy.deepcopy(history), None, other))
    history.append({"role": "tool", "content": "file contents"})
    assert _session_prompt_ids(history, tokenizer, tools=None, session=session) == _render_token_ids(
        history, tokenizer, tools=None
    )


def test_responses_http_roundtrip_trains_original_thinking_and_tools_in_one_sample(tokenizer):
    async def run():
        raw = (
            "Check both paths.\n</think>\n\nInspecting.\n\n"
            "<tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call><|im_end|>"
        )
        outputs = [
            tokenizer.encode(text, add_special_tokens=False) for text in [raw, "Done.\n</think>\n\nFixed.<|im_end|>"]
        ]
        captured = []
        async with FakeSGLangServer([[(-0.1, token) for token in output] for output in outputs]) as upstream:
            adapter = ResponsesAdapter(
                tokenizer=tokenizer,
                sglang_url=upstream.url,
                tool_parser="qwen3_coder",
                reasoning_parser="qwen3",
                debug_callback=lambda sid, messages, tools, reply, turn: captured.append(asdict(turn)),
            )
            adapter.open_session("sid")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            body = {
                "input": [{"role": "user", "content": "Read the files."}],
                "store": False,
                "tools": [
                    {
                        "type": "function",
                        "name": "read",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    }
                ],
            }
            try:
                response = await client.post("/v1/responses", headers={"Authorization": "Bearer sid"}, json=body)
                assert response.status == 200, await response.text()
                first = await response.json()
                call = next(item for item in first["output"] if item["type"] == "function_call")
                assert all(item["type"] != "reasoning" for item in first["output"])
                body["input"] += first["output"] + [
                    {"type": "function_call_output", "call_id": call["call_id"], "output": "file contents"}
                ]
                response = await client.post("/v1/responses", headers={"Authorization": "Bearer sid"}, json=body)
                assert response.status == 200, await response.text()
                samples = await adapter.finish_session("sid", base_sample=Sample(index=0), reward=1)
            finally:
                await client.close()
        prefix = upstream.requests[0]["input_ids"] + outputs[0]
        assert upstream.requests[1]["input_ids"][: len(prefix)] == prefix
        assert len(samples) == 1
        assert audit_token_records(samples, captured)["every_sampled_token_retained_once"]

    asyncio.run(run())


def test_anthropic_http_roundtrip_merges_thinking_and_tool_results(tokenizer):
    async def run():
        outputs = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in [
                "Inspect the file.\n</think>\n\nReading.\n\n<tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call><|im_end|>",
                "Finished checking.\n</think>\n\nDone.<|im_end|>",
            ]
        ]
        captured = []
        async with FakeSGLangServer([[(-0.1, token) for token in output] for output in outputs]) as upstream:
            adapter = AnthropicAdapter(
                tokenizer=tokenizer,
                sglang_url=upstream.url,
                tool_parser="qwen3_coder",
                reasoning_parser="qwen3",
                debug_callback=lambda sid, messages, tools, reply, turn: captured.append(asdict(turn)),
            )
            adapter.open_session("sid")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            body = {
                "messages": [{"role": "user", "content": "Read the file."}],
                "tools": [
                    {"name": "read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
                ],
            }
            try:
                response = await client.post("/v1/messages", headers={"x-api-key": "sid"}, json=body)
                assert response.status == 200, await response.text()
                first = await response.json()
                call = next(block for block in first["content"] if block["type"] == "tool_use")
                assert any(block["type"] == "thinking" for block in first["content"])
                body["messages"] += [
                    {"role": "assistant", "content": first["content"]},
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": "file contents"}],
                    },
                ]
                response = await client.post("/v1/messages", headers={"x-api-key": "sid"}, json=body)
                assert response.status == 200, await response.text()
                samples = await adapter.finish_session("sid", base_sample=Sample(index=0), reward=1)
            finally:
                await client.close()
        prefix = upstream.requests[0]["input_ids"] + outputs[0]
        assert upstream.requests[1]["input_ids"][: len(prefix)] == prefix
        assert len(samples) == 1
        assert audit_token_records(samples, captured)["every_sampled_token_retained_once"]

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
