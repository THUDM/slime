import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout import sglang_rollout

NUM_GPUS = 0


def test_generate_state_uses_each_calls_refill_arguments(monkeypatch):
    state = object.__new__(sglang_rollout.GenerateState)
    state.pendings = set()
    state.remaining_batch_size = 0
    state.sampling_params = {}
    seen = []

    async def generate(args, group, **_kwargs):
        seen.append((args.rollout_batch_size, group))
        return group

    monkeypatch.setattr(sglang_rollout, "generate_and_rm_group", generate)

    async def run_calls():
        state.submit_generate_tasks(SimpleNamespace(rollout_batch_size=8), [["initial"]])
        await asyncio.gather(*state.pendings)
        state.pendings.clear()
        state.submit_generate_tasks(SimpleNamespace(rollout_batch_size=2), [["replacement"]])
        await asyncio.gather(*state.pendings)

    asyncio.run(run_calls())

    assert seen == [(8, ["initial"]), (2, ["replacement"])]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
