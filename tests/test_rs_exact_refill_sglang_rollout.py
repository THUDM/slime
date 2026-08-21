import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout import sglang_rollout

NUM_GPUS = 0


def _make_state(monkeypatch, default_batch_size):
    args = SimpleNamespace(
        hf_checkpoint="checkpoint",
        n_samples_per_prompt=1,
        rollout_max_response_len=128,
        rollout_skip_special_tokens=False,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_temperature=1.0,
        rollout_top_k=-1,
        rollout_top_p=1.0,
        sglang_dp_size=1,
        sglang_enable_deterministic_inference=False,
        sglang_server_concurrency=1,
    )
    monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(sglang_rollout, "load_processor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sglang_rollout, "get_rollout_num_engines", lambda _args: 1)
    state = object.__new__(sglang_rollout.GenerateState)
    state.__init__(args)
    state.args.rollout_batch_size = default_batch_size
    return state


def test_generate_state_preserves_legacy_submit_call(monkeypatch):
    state = _make_state(monkeypatch, default_batch_size=8)
    seen = []

    async def generate(args, group, **_kwargs):
        seen.append((args.rollout_batch_size, group))
        return group

    monkeypatch.setattr(sglang_rollout, "generate_and_rm_group", generate)

    async def run_call():
        state.submit_generate_tasks([["initial"]])
        await asyncio.gather(*state.pendings)

    asyncio.run(run_call())

    assert state.args.rollout_batch_size == 8
    assert seen == [(8, ["initial"])]


def test_generate_state_uses_each_calls_refill_argument_override(monkeypatch):
    state = _make_state(monkeypatch, default_batch_size=8)
    seen = []

    async def generate(args, group, **_kwargs):
        seen.append((args.rollout_batch_size, group))
        return group

    monkeypatch.setattr(sglang_rollout, "generate_and_rm_group", generate)

    async def run_calls():
        state.submit_generate_tasks([["initial"]])
        await asyncio.gather(*state.pendings)
        state.pendings.clear()
        state.submit_generate_tasks([["replacement"]], args=SimpleNamespace(rollout_batch_size=2))
        await asyncio.gather(*state.pendings)

    asyncio.run(run_calls())

    assert seen == [(8, ["initial"]), (2, ["replacement"])]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
