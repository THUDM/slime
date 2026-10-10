"""Verify model token provenance through branching and reject broken captures."""

import asyncio
import copy
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_e2e_helpers import audit_token_records  # noqa: E402
from tests.test_agent._fakes import FakeTokenizer  # noqa: E402

from slime.agent.adapters.common import Session, call_sglang_generate  # noqa: E402
from slime.agent.adapters.openai import OpenAIAdapter  # noqa: E402
from slime.agent.trajectory import TrajectoryManager, TurnRecord  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


def branched_trajectory():
    manager = TrajectoryManager(fork_threshold_tokens=0)
    user = {"role": "user", "content": "fix it"}
    assistant = {"role": "assistant", "content": "read files"}
    turns = [
        TurnRecord([1, 2], [10, 11], "stop", [-0.1, -0.2]),
        TurnRecord([1, 2, 10, 11, 3, 4], [20, 21], "stop", [-0.3, -0.4]),
        TurnRecord([1, 2, 10, 11, 5, 6], [30], "stop", [-0.5]),
        TurnRecord([1, 2, 99, 8], [40], "stop", [-0.6]),
    ]
    histories = [
        [user],
        [user, assistant, {"role": "tool", "content": "file A"}],
        [user, assistant, {"role": "tool", "content": "file B"}],
        [user, {"role": "assistant", "content": "rewritten history"}],
    ]
    for i, (turn, history) in enumerate(zip(turns, histories, strict=True)):
        manager.record_turn(
            "agent",
            turn=turn,
            prompt_messages=history,
            response_message=assistant if i == 0 else {"role": "assistant", "content": f"branch {i}"},
        )
    samples = manager.get_trajectory("agent", base_sample=Sample(index=11, group_index=7), reward=1)
    return samples, [asdict(turn) for turn in turns]


def test_shared_prefix_and_rewritten_history_keep_exact_sampled_tokens_once():
    samples, turns = branched_trajectory()
    report = audit_token_records(samples, turns)
    assert len(samples) == 3
    assert report["sampled_tokens"] == 6
    assert report["every_sampled_token_retained_once"]
    assert {(s.group_index, s.index, s.rollout_id, s.reward) for s in samples} == {(7, 11, 11, 1)}


@pytest.mark.parametrize("corruption", ["token", "prompt", "logprob", "missing", "duplicate"])
def test_e2e_audit_detects_corrupted_or_duplicate_training_spans(corruption):
    samples, turns = branched_trajectory()
    if corruption == "token":
        samples[-1].tokens[-1] += 100
    elif corruption == "prompt":
        samples[-1].tokens[0] += 100
    elif corruption == "logprob":
        samples[-1].rollout_log_probs[-1] -= 0.1
    elif corruption == "missing":
        samples.pop()
    else:
        samples.append(copy.deepcopy(samples[-1]))
    with pytest.raises(AssertionError):
        audit_token_records(samples, turns)


@pytest.mark.parametrize("corruption", [None, "missing", "count", "ids", "nonfinite"])
def test_model_wire_uses_token_ids_and_requires_complete_logprobs(corruption):
    async def run():
        async def reply(request):
            payload = await request.json()
            assert payload["input_ids"] == [51, 52] and "text" not in payload
            assert payload["return_logprob"] is True
            result = {
                "text": "This text deliberately does not encode to the sampled IDs.",
                "output_ids": [101, 102],
                "meta_info": {"output_token_logprobs": [[-0.1, 101], [-0.2, 102]], "completion_tokens": 2},
            }
            if corruption == "missing":
                del result["meta_info"]["output_token_logprobs"]
            elif corruption == "count":
                result["meta_info"]["completion_tokens"] = 3
            elif corruption == "ids":
                result["output_ids"][0] = 999
            elif corruption == "nonfinite":
                result["meta_info"]["output_token_logprobs"][0][0] = float("nan")
            return web.json_response(result)

        app = web.Application()
        app.router.add_post("/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(tokenizer=FakeTokenizer(), sglang_url=str(server.make_url("")).rstrip("/"))
            if corruption:
                with pytest.raises(ValueError, match="SGLang"):
                    await call_sglang_generate([51, 52], Session(), {}, adapter=adapter)
            else:
                turn = await call_sglang_generate([51, 52], Session(), {}, adapter=adapter)
                assert turn.prompt_ids == [51, 52]
                assert turn.output_ids == [101, 102]
                assert turn.output_log_probs == [-0.1, -0.2]

    asyncio.run(run())


@pytest.mark.parametrize(
    "algorithm,topk,steps,drafts,rollout_limit,server_limit,prompt_len,expected",
    [
        (None, 1, 3, 4, 16, 16, 6, 10),
        ("EAGLE", 1, 3, 4, 16, 16, 6, 6),
        ("EAGLE3", 2, 3, 4, 16, 16, 6, 4),
        ("EAGLE", 1, 3, 4, 10, 20, 6, 4),
        ("EAGLE", 1, 3, 4, 16, None, 6, 6),
        ("EAGLE", 1, 3, 4, 0, 16, 6, 6),
        ("EAGLE", 1, 3, 4, 16, 16, 12, 0),
        ("EAGLE", 1, 3, 4, 16, 16, 13, 0),
        ("EAGLE", 1, 3, 4, 4, 4, 1, 0),
    ],
)
def test_generation_budget_reserves_speculative_slots(
    algorithm, topk, steps, drafts, rollout_limit, server_limit, prompt_len, expected
):
    async def run():
        requests = []

        async def reply(request):
            payload = await request.json()
            requests.append(payload)
            assert payload["sampling_params"]["max_new_tokens"] == expected
            return web.json_response({"meta_info": {"output_token_logprobs": [[-0.1, 101]]}})

        app = web.Application()
        app.router.add_post("/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(
                tokenizer=FakeTokenizer(),
                sglang_url=str(server.make_url("")).rstrip("/"),
                rollout_args=SimpleNamespace(
                    sglang_context_length=server_limit,
                    sglang_speculative_algorithm=algorithm,
                    sglang_speculative_eagle_topk=topk,
                    sglang_speculative_num_steps=steps,
                    sglang_speculative_num_draft_tokens=drafts,
                ),
            )
            turn = await call_sglang_generate(
                [51] * prompt_len, Session(max_context_tokens=rollout_limit), {}, adapter=adapter
            )
            assert bool(requests) == bool(expected)
            assert turn.output_ids == ([101] if expected else [])
            if not expected:
                assert turn.finish_reason == "length"

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
