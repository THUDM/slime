import asyncio
import types

import pytest

from slime.rollout.sample_hooks import apply_rollout_sample_hooks, rollout_context, set_current_rollout_id
from slime.utils.types import Sample

NUM_GPUS = 0


def sync_hook(args, sample, *, rollout_id=None):
    sample.metadata["sync_hook"] = (args.marker, rollout_id)


async def async_hook(args, sample, *, evaluation=False):
    sample.metadata["async_hook"] = evaluation
    return sample


def invalid_hook(args, sample):
    return {"sample": sample}


@pytest.mark.unit
def test_rollout_sample_hooks_preserve_nested_shape_and_filter_kwargs():
    args = types.SimpleNamespace(
        marker="seen",
        rollout_sample_hook_path=[f"{__name__}.sync_hook", f"{__name__}.async_hook"],
    )
    samples = [[Sample(index=0, metadata={})], [Sample(index=1, metadata={})]]
    set_current_rollout_id(7)

    result = asyncio.run(apply_rollout_sample_hooks(args, samples, evaluation=True, ignored="value"))

    assert result is not samples
    assert [[sample.index for sample in group] for group in result] == [[0], [1]]
    for group in result:
        assert group[0].metadata == {"sync_hook": ("seen", 7), "async_hook": True}


@pytest.mark.unit
def test_rollout_sample_hook_rejects_invalid_return_type():
    args = types.SimpleNamespace(rollout_sample_hook_path=[f"{__name__}.invalid_hook"])

    with pytest.raises(TypeError, match="expected Sample or None"):
        asyncio.run(apply_rollout_sample_hooks(args, Sample(index=0)))


@pytest.mark.unit
def test_rollout_sample_hooks_are_noop_when_unconfigured():
    args = types.SimpleNamespace(rollout_sample_hook_path=[])
    sample = Sample(index=0)

    assert asyncio.run(apply_rollout_sample_hooks(args, sample)) is sample


def test_concurrent_groups_keep_their_own_rollout_hook_context():
    args = types.SimpleNamespace(marker="seen", rollout_sample_hook_path=[f"{__name__}.sync_hook"])

    async def generate(rollout_id):
        with rollout_context(rollout_id):
            await asyncio.sleep(0)
            sample = await apply_rollout_sample_hooks(args, Sample(index=rollout_id))
            return sample.metadata["sync_hook"]

    async def exercise():
        assert await asyncio.gather(generate(10), generate(11)) == [("seen", 10), ("seen", 11)]

    asyncio.run(exercise())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
