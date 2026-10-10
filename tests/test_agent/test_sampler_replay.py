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
    first = capture(top_p, sc, r3)
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
    first = capture(0.95, True, False)
    second = replace(
        first,
        prompt_ids=first.prompt_ids + first.output_ids + [9, 8, 7],
        output_ids=[103, 104, 105],
        output_log_probs=[-0.1, -0.2, -0.3],
        replay={
            "rollout_top_p_token_ids": torch.tensor([103, 104, 204, 105], dtype=torch.int32),
            "rollout_top_p_token_offsets": torch.tensor([0, 1, 3, 4], dtype=torch.int32),
            "rollout_top_p_log_probs": torch.tensor([-0.1, -0.2, -1.0, -0.3]),
        },
    )
    third = replace(second, prompt_ids=first.prompt_ids + first.output_ids + [6, 5])
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
    turns = [asdict(t) for t in (first, second, third)]
    report = audit_token_records(samples, turns)
    assert report["sampled_tokens"] == 8
    samples[0].rollout_top_p_log_probs[-1] -= 0.5
    with pytest.raises(ValueError, match="sampler distribution must include the sampled token"):
        audit_token_records(samples, turns)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
