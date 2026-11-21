"""Agent adapters preserve the sampler's top-p, SC and R3 snapshots."""

import asyncio
import base64
import copy
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestServer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_e2e_helpers import audit_token_records
from tests.test_agent._fakes import FakeTokenizer

from slime.agent.adapters.common import Session, call_sglang_generate
from slime.agent.adapters.openai import OpenAIAdapter
from slime.agent.trajectory import TrajectoryManager
from slime.observability.rollout_metrics import _compute_prefix_cache_metrics, _compute_spec_metrics
from slime.observability.trace_utils import bind_trace
from slime.utils.types import Sample

NUM_GPUS = 0


def encoded(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode()


def capture(top_p, sc, r3, missing=None):
    async def run():
        args = SimpleNamespace(
            rollout_top_p=top_p,
            rollout_temperature=1.0,
            use_score_centering=sc,
            score_centering_top_k=2,
            use_rollout_routing_replay=r3,
            num_layers=2,
            moe_router_topk=2,
        )

        async def reply(request):
            payload = await request.json()
            assert payload["input_ids"] == [51, 52, 53]
            params = payload["sampling_params"]
            assert params["top_p"] == top_p and params.get("top_k", -1) == -1
            assert payload.get("return_routed_experts", False) == r3
            ids = [101, 201, 102, 202]
            logps = np.log([0.7, 0.3, 0.6, 0.4]).astype(np.float32)
            meta = {
                "completion_tokens": 2,
                "output_token_logprobs": [[float(logps[0]), 101], [float(logps[2]), 102]],
            }
            if top_p < 1:
                assert params["custom_params"]["return_top_p_token_ids"]
                meta.update(top_p_token_ids=encoded(ids, "<i4"), top_p_token_offsets=encoded([0, 2, 4], "<i4"))
                if sc:
                    assert params["custom_params"]["return_top_p_log_probs"]
                    meta["top_p_log_probs"] = encoded(logps, "<f4")
            elif sc:
                assert payload["top_logprobs_num"] == 2
                meta.update(
                    output_topk_token_ids=encoded(ids, "<i4"),
                    output_topk_log_probs=encoded(logps, "<f4"),
                    output_topk_shape=[2, 2],
                )
            if r3:
                meta["routed_experts"] = encoded(np.arange(16) % 4, "<i4")
            if missing == "nucleus":
                del meta["top_p_token_ids"], meta["top_p_token_offsets"]
            elif missing == "probabilities":
                del meta["top_p_log_probs"]
            elif missing == "routes":
                del meta["routed_experts"]
            return web.json_response({"meta_info": meta})

        app = web.Application()
        app.router.add_post("/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(
                tokenizer=FakeTokenizer(), sglang_url=str(server.make_url("")).rstrip("/"), rollout_args=args
            )
            return await call_sglang_generate(
                [51, 52, 53], Session(sampling_defaults={"top_p": top_p, "temperature": 1.0}), {}, adapter=adapter
            )

    return asyncio.run(run())


@pytest.mark.parametrize(
    "top_p,sc,r3",
    [(0.95, False, False), (0.95, True, False), (1.0, True, False), (0.95, True, True), (1.0, True, True)],
)
@pytest.mark.parametrize("truncate", [False, True])
def test_replay_survives_exact_prefix_extensions_and_context_limit(top_p, sc, r3, truncate):
    first = replace(
        capture(top_p, sc, r3),
        spec_info=Sample.SpecInfo(1, 3, 1, 2),
        prefix_cache_info=Sample.PrefixCacheInfo(1, 3),
    )
    replay = copy.deepcopy(first.replay)
    prompt = first.prompt_ids + first.output_ids + [9]
    if r3:
        replay["rollout_routed_experts"] = torch.arange(28, dtype=torch.int32).reshape(7, 2, 2) % 4
    second = replace(first, prompt_ids=prompt, replay=replay)
    manager = TrajectoryManager(fork_threshold_tokens=1024)
    user = {"role": "user", "content": "repair"}
    assistant = {"role": "assistant", "content": "read"}
    manager.record_turn("s", turn=first, prompt_messages=[user], response_message=assistant)
    manager.record_turn(
        "s",
        turn=second,
        prompt_messages=[user, assistant, {"role": "tool", "content": "files"}],
        response_message={"role": "assistant", "content": "fixed"},
    )
    samples = manager.get_trajectory(
        "s", base_sample=Sample(index=4, group_index=2), reward=1, max_sample_tokens=7 if truncate else 0
    )
    assert len(samples) == (2 if r3 else 1)
    assert {(s.index, s.rollout_id, s.group_index, s.reward) for s in samples} == {(4, 4, 2, 1)}
    assert sum(s.spec_info.spec_accept_token_num for s in samples) == 2
    assert sum(s.spec_info.spec_draft_token_num for s in samples) == 6
    assert sum(s.spec_info.completion_token_num for s in samples) == 4
    assert sum(s.prefix_cache_info.total_prompt_tokens for s in samples) == 6
    if not truncate:
        report = audit_token_records(samples, [asdict(first), asdict(second)])
        assert set(report["replay_metadata_fields_verified"]) == set(replay)
    else:
        sample = samples[-1]
        assert sample.response_length == (1 if r3 else 4) and sample.tokens == prompt + [101]
        assert sample.loss_mask == ([1] if r3 else [1, 1, 0, 1])
        if top_p < 1:
            assert sample.rollout_top_p_token_offsets.tolist() == ([0, 2] if r3 else [0, 2, 4, 4, 6])
            assert sample.rollout_top_p_token_ids.tolist() == ([101, 201] if r3 else [101, 201, 102, 202, 101, 201])
            if sc:
                assert len(sample.rollout_top_p_log_probs) == (2 if r3 else 6)
        else:
            assert sample.rollout_topk_token_ids.tolist() == (
                [[101, 201]] if r3 else [[101, 201], [102, 202], [0, 1], [101, 201]]
            )
        if r3:
            assert sample.rollout_routed_experts.shape == (6, 2, 2)


@pytest.mark.parametrize("missing", ["nucleus", "probabilities", "routes"])
def test_missing_requested_replay_metadata_fails_closed(missing):
    with pytest.raises(ValueError):
        capture(0.95, True, True, missing)


def test_merged_ragged_distributions_follow_tokens_and_shared_branches_train_once():
    first = replace(
        capture(0.95, True, False),
        spec_info=Sample.SpecInfo(1, 3, 1, 2),
        prefix_cache_info=Sample.PrefixCacheInfo(1, 3),
    )
    second = replace(
        first,
        prompt_ids=first.prompt_ids + first.output_ids + [9, 8, 7],
        output_ids=[103, 104, 105],
        output_log_probs=[-0.1, -0.2, -0.3],
        spec_info=Sample.SpecInfo(2, 6, 2, 3),
        prefix_cache_info=Sample.PrefixCacheInfo(4, 8),
        replay={
            "rollout_top_p_token_ids": torch.tensor([103, 104, 204, 105], dtype=torch.int32),
            "rollout_top_p_token_offsets": torch.tensor([0, 1, 3, 4], dtype=torch.int32),
            "rollout_top_p_log_probs": torch.tensor([-0.1, -0.2, -1.0, -0.3]),
        },
    )
    third = replace(
        second,
        prompt_ids=first.prompt_ids + first.output_ids + [6, 5],
        prefix_cache_info=Sample.PrefixCacheInfo(4, 7),
    )
    user, assistant = {"role": "user", "content": "repair"}, {"role": "assistant", "content": "inspect"}
    manager = TrajectoryManager(fork_threshold_tokens=0)
    manager.record_turn("s", turn=first, prompt_messages=[user], response_message=assistant)
    for turn, branch in [(second, "left"), (third, "right")]:
        manager.record_turn(
            "s",
            turn=turn,
            prompt_messages=[user, assistant, {"role": "tool", "content": branch}],
            response_message={"role": "assistant", "content": "fixed"},
        )
    samples = manager.get_trajectory("s", base_sample=Sample(index=4, group_index=2), reward=1)
    assert len(samples) == 2
    assert samples[0].loss_mask == [1, 1, 0, 0, 0, 1, 1, 1]
    assert samples[1].loss_mask == [0, 0, 0, 0, 1, 1, 1]
    assert samples[1].rollout_log_probs[:2] == first.output_log_probs
    assert samples[0].rollout_top_p_token_offsets.tolist() == [0, 2, 4, 4, 4, 4, 5, 7, 8]
    assert samples[0].rollout_top_p_token_ids.tolist() == [101, 201, 102, 202, 103, 104, 204, 105]
    assert [s.spec_info for s in samples] == [Sample.SpecInfo(3, 9, 3, 5), Sample.SpecInfo(2, 6, 2, 3)]
    assert _compute_spec_metrics(SimpleNamespace(sglang_speculative_algorithm="EAGLE"), samples) == pytest.approx(
        {"spec_accept_rate": 1 / 3, "spec_accept_length": 19 / 12}
    )
    assert _compute_prefix_cache_metrics(samples)["prefix_cache_hit_rate"] == pytest.approx(9 / 18)
    # A serialized worker handoff must retain the counters as well as tokens.
    assert Sample.from_dict(samples[0].to_dict()).spec_info == samples[0].spec_info
    turns = [asdict(t) for t in (first, second, third)]
    report = audit_token_records(samples, turns)
    assert report["sampled_tokens"] == 8
    samples[0].rollout_top_p_log_probs[-1] -= 0.5
    with pytest.raises(ValueError, match="sampler distribution must include the sampled token"):
        audit_token_records(samples, turns)


@pytest.mark.parametrize("top_p,r3", [(0.95, False), (1.0, False), (0.95, True)])
def test_partial_turn_continues_exact_tokens_and_preserves_each_policy(top_p, r3):
    async def run():
        args = SimpleNamespace(
            partial_rollout=True,
            sglang_speculative_algorithm="EAGLE",
            rollout_top_p=top_p,
            rollout_temperature=1.0,
            use_score_centering=True,
            score_centering_top_k=2,
            use_rollout_routing_replay=r3,
            num_layers=2,
            moe_router_topk=2,
        )
        requests = []
        resume = asyncio.Event()
        waiting = asyncio.Event()

        async def reply(request):
            payload = await request.json()
            requests.append(payload)
            index = len(requests) - 1
            assert request.headers["X-SMG-Routing-Key"] == "kept-sandbox"
            assert payload["input_ids"] == [51, 52, 53] + [101] * (index == 2)
            assert payload["sampling_params"]["max_new_tokens"] == (3 if index < 2 else 2)
            assert len({r["rid"] for r in requests}) == len(requests)
            if index == 2:
                waiting.set()
                await resume.wait()
            tokens = [] if index == 0 else [100 + index]
            meta = {
                "finish_reason": {"type": "stop" if index == 2 else "abort"},
                "completion_tokens": len(tokens),
                "output_token_logprobs": [[float(np.log(0.7)), t] for t in tokens],
                "weight_version": str(index),
                "spec_num_correct_drafts": int(index == 1),
                "spec_num_proposed_drafts": 3 if tokens else 0,
                "spec_verify_ct": int(bool(tokens)),
                "prompt_tokens": len(payload["input_ids"]),
                "cached_tokens": index,
            }
            if tokens:
                ids, logps = [tokens[0], 200 + index], np.log([0.7, 0.3])
                if top_p < 1:
                    meta.update(
                        top_p_token_ids=encoded(ids, "<i4"),
                        top_p_token_offsets=encoded([0, 2], "<i4"),
                        top_p_log_probs=encoded(logps, "<f4"),
                    )
                else:
                    meta.update(
                        output_topk_token_ids=encoded(ids, "<i4"),
                        output_topk_log_probs=encoded(logps, "<f4"),
                        output_topk_shape=[1, 2],
                    )
                if r3:
                    assert payload.get("routed_experts_start_len", 0) == (3 if index == 2 else 0)
                    meta["routed_experts"] = encoded([index] * (4 if index == 2 else 12), "<i4")
            return web.json_response({"output_ids": tokens, "meta_info": meta})

        app = web.Application()
        app.router.add_post("/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(tokenizer=FakeTokenizer(), sglang_url=str(server.make_url("")), rollout_args=args)
            task = asyncio.create_task(
                call_sglang_generate(
                    [51, 52, 53],
                    Session(sampling_defaults={"top_p": top_p, "max_new_tokens": 3}),
                    {},
                    adapter=adapter,
                    session_id="kept-sandbox",
                )
            )
            try:
                await asyncio.wait_for(waiting.wait(), 5)
                assert not task.done(), "The CLI must not receive an unfinished model turn"
            finally:
                resume.set()
            turn = await asyncio.wait_for(task, 5)
        assert turn.output_ids == [101, 102] and turn.finish_reason == "stop"
        assert turn.output_log_probs == pytest.approx([np.log(0.7)] * 2)
        assert turn.weight_versions == ["1", "2"]
        assert turn.spec_info == Sample.SpecInfo(1, 6, 2, 2)
        assert turn.prefix_cache_info == Sample.PrefixCacheInfo(3, 10)
        if top_p < 1:
            assert turn.replay["rollout_top_p_token_ids"].tolist() == [101, 201, 102, 202]
            assert turn.replay["rollout_top_p_token_offsets"].tolist() == [0, 2, 4]
        else:
            assert turn.replay["rollout_topk_token_ids"].tolist() == [[101, 201], [102, 202]]
        if r3:
            assert turn.replay["rollout_routed_experts"].tolist() == [[[1, 1], [1, 1]]] * 3 + [[[2, 2], [2, 2]]]
        manager = TrajectoryManager()
        manager.record_turn(
            "s",
            turn=turn,
            prompt_messages=[{"role": "user", "content": "fix"}],
            response_message={"role": "assistant", "content": "fixed"},
        )
        samples = manager.get_trajectory("s", base_sample=Sample(index=0), reward=1)
        assert samples[0].weight_versions == ["1", "2"]
        assert samples[0].loss_mask == [1, 1]
        assert _compute_spec_metrics(args, samples) == pytest.approx(
            {"spec_accept_rate": 1 / 6, "spec_accept_length": 1}
        )
        assert _compute_prefix_cache_metrics(samples)["prefix_cache_hit_rate"] == pytest.approx(3 / 10)
        audit_token_records(samples, [asdict(turn)])

    asyncio.run(run())


@pytest.mark.parametrize(
    "partial,budget,expected_calls,finish", [(True, 2, 2, "stop"), (True, 1, 1, "length"), (False, 2, 1, "abort")]
)
def test_partial_turn_budget_and_opt_in(partial, budget, expected_calls, finish):
    async def run():
        calls = []

        async def reply(request):
            assert request.headers["X-SMG-Routing-Key"] == "agent-trace-test"
            calls.append(await request.json())
            return web.json_response(
                {
                    "meta_info": {
                        "finish_reason": {"type": "abort" if len(calls) == 1 else "stop"},
                        "completion_tokens": 1,
                        "output_token_logprobs": [[-0.5, 101]],
                    }
                }
            )

        app = web.Application()
        app.router.add_post("/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(
                tokenizer=FakeTokenizer(),
                sglang_url=str(server.make_url("")),
                rollout_args=SimpleNamespace(partial_rollout=partial),
            )
            sample = Sample(index=0)
            turn = await call_sglang_generate(
                [51],
                Session(sampling_defaults={"max_new_tokens": budget}, trace=bind_trace(sample)),
                {},
                adapter=adapter,
                session_id="agent-trace-test",
            )
        assert len(calls) == expected_calls and turn.finish_reason == finish
        assert len(turn.output_ids) == expected_calls
        events = sample.trace["events"]
        starts = [event for event in events if event["type"] == "span_start"]
        ends = [event for event in events if event["type"] == "span_end"]
        assert len(starts) == len(ends) == expected_calls
        assert [event["attrs"]["continuation_tokens"] for event in starts] == list(range(expected_calls))
        assert sum(event["attrs"]["completion_tokens"] for event in ends) == expected_calls
        assert all(event["attrs"]["http_status"] == 200 for event in ends)

    asyncio.run(run())


@pytest.mark.parametrize("error", ["http", "missing_replay"])
def test_partial_turn_does_not_retry_errors_or_train_incomplete_replay(error):
    async def run():
        calls = []

        async def reply(request):
            calls.append(await request.json())
            if len(calls) == 2 and error == "http":
                return web.Response(status=503, text="model unavailable")
            meta = {
                "finish_reason": {"type": "abort" if len(calls) == 1 else "stop"},
                "output_token_logprobs": [[float(np.log(0.7)), 101]],
            }
            if len(calls) == 1:
                meta.update(
                    top_p_token_ids=encoded([101, 201], "<i4"),
                    top_p_token_offsets=encoded([0, 2], "<i4"),
                )
            return web.json_response({"meta_info": meta})

        app = web.Application()
        app.router.add_post("/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(
                tokenizer=FakeTokenizer(),
                sglang_url=str(server.make_url("")),
                rollout_args=SimpleNamespace(partial_rollout=True),
            )
            with pytest.raises(RuntimeError if error == "http" else ValueError):
                await call_sglang_generate([51], Session(sampling_defaults={"top_p": 0.95}), {}, adapter=adapter)
        assert len(calls) == 2

    asyncio.run(run())


def test_partial_turn_cancel_aborts_current_continuation():
    async def run():
        calls, aborted = [], []
        waiting, release = asyncio.Event(), asyncio.Event()

        async def reply(request):
            calls.append(await request.json())
            if len(calls) == 2:
                waiting.set()
                await release.wait()
            return web.json_response(
                {
                    "meta_info": {
                        "finish_reason": {"type": "abort"},
                        "output_token_logprobs": [[-0.5, 101]],
                    }
                }
            )

        async def abort(request):
            aborted.append((await request.json())["rid"])
            release.set()
            return web.json_response({"ok": True})

        app = web.Application()
        app.router.add_post("/generate", reply)
        app.router.add_post("/abort_request", abort)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(
                tokenizer=FakeTokenizer(),
                sglang_url=str(server.make_url("")),
                rollout_args=SimpleNamespace(partial_rollout=True),
            )
            task = asyncio.create_task(call_sglang_generate([51], Session(), {}, adapter=adapter))
            try:
                await asyncio.wait_for(waiting.wait(), 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert aborted == [calls[1]["rid"]]
            finally:
                release.set()
                task.cancel()

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
