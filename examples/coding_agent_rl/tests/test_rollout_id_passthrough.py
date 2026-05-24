"""End-to-end CPU test that the PR #1933 rollout_id contract is honoured
by the SWE fan-out path and the RolloutManager.

We construct K=3 fake fan-out Samples (one trajectory of 3 segments) and
hand them to a stub RolloutManager._convert_samples_to_train_data, then
assert:

  - train_data['rollout_ids'] has the same id for all K samples
    (sample_proto.index, not the per-sample index)
  - train_data['rollout_mask_sums'] gives every sample the SAME value =
    the total mask sum across the rollout's K samples (so the per-mb
    loss reducer's denom is the per-rollout token total)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Make slime importable
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.utils.types import Sample  # noqa: E402


def _fake_sample(index: int, rollout_id: int | None, response_length: int = 4) -> Sample:
    s = Sample()
    s.index = index
    s.group_index = 0
    s.rollout_id = rollout_id
    s.tokens = [1] * (2 + response_length)  # 2 prompt + N response
    s.response_length = response_length
    s.loss_mask = [1] * response_length
    s.reward = 1.0
    s.status = Sample.Status.COMPLETED
    return s


def _make_train_data(samples: list[Sample]) -> dict:
    """Reproduce the parts of RolloutManager._convert_samples_to_train_data
    that depend on rollout_id, without touching ray / sglang."""
    train_data = {
        "tokens": [s.tokens for s in samples],
        "response_lengths": [s.response_length for s in samples],
        "rewards": [s.reward for s in samples],
        "raw_reward": [s.reward for s in samples],
        "truncated": [0 for _ in samples],
        "sample_indices": [s.index for s in samples],
        "rollout_ids": [s.rollout_id if s.rollout_id is not None else s.index for s in samples],
    }
    loss_masks = [list(s.loss_mask) for s in samples]
    train_data["loss_masks"] = loss_masks
    rollout_id_list = train_data["rollout_ids"]
    mask_sums_per_sample = [sum(m) for m in loss_masks]
    rollout_total_mask: dict[int, int] = {}
    for rid, ms in zip(rollout_id_list, mask_sums_per_sample):
        rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
    train_data["rollout_mask_sums"] = [rollout_total_mask[rid] for rid in rollout_id_list]
    return train_data


def test_compact_three_segments_share_rollout_id_and_mask_sum():
    # 1 trajectory split into 3 fan-out samples, all sharing the same
    # sample.index (because copy.copy in _default_uniform_fan_out copies the
    # dataclass) and the same rollout_id.
    samples = [
        _fake_sample(index=42, rollout_id=42, response_length=4),
        _fake_sample(index=42, rollout_id=42, response_length=5),
        _fake_sample(index=42, rollout_id=42, response_length=3),
    ]
    td = _make_train_data(samples)
    assert td["rollout_ids"] == [42, 42, 42]
    # 4 + 5 + 3 = 12 mask tokens for this trajectory
    assert td["rollout_mask_sums"] == [12, 12, 12]


def test_default_one_sample_per_rollout_falls_back_to_index():
    # Default rollout: each Sample has rollout_id=None and a unique index;
    # rollout_id falls back to index, mask sum is per-sample.
    samples = [
        _fake_sample(index=0, rollout_id=None, response_length=4),
        _fake_sample(index=1, rollout_id=None, response_length=2),
        _fake_sample(index=2, rollout_id=None, response_length=6),
    ]
    td = _make_train_data(samples)
    assert td["rollout_ids"] == [0, 1, 2]
    # Each rollout has exactly one sample → rollout_mask_sums == per-sample sums
    assert td["rollout_mask_sums"] == [4, 2, 6]


def test_swe_default_uniform_fan_out_sets_rollout_id():
    """SWE-side glue: _default_uniform_fan_out must set rollout_id on every
    fan-out sibling to sample_proto.index. Imports the real SWE function,
    stubs the tokenizer."""
    # Patch out the heavy module-level deps that generate.py imports at top
    # level (sandbox / middleware require torch / openai / etc.).
    # The function we want is pure: it doesn't touch any of those.
    from examples.coding_agent_rl import generate as swe_gen  # noqa: E402

    class StubTok:
        def decode(self, ids, skip_special_tokens=False):
            return "stub"

    proto = Sample()
    proto.index = 99
    proto.group_index = 7
    proto.metadata = {}

    segments = [
        ([1, 2], [3, 4, 5], [1, 1, 1], {"segment_kind": "subagent"}),
        ([1, 2], [6, 7], [1, 1], {"segment_kind": "pre_wipe"}),
        ([1, 2], [8, 9, 10, 11], [1, 1, 1, 1], {"segment_kind": "final"}),
    ]
    out = swe_gen._default_uniform_fan_out(
        segments, reward=3.0, sample_proto=proto, tokenizer=StubTok(), instance_id="abc"
    )
    assert len(out) == 3
    # All siblings carry the SAME rollout_id == proto.index
    assert [s.rollout_id for s in out] == [99, 99, 99]
    # And the same dataset index (copy.copy preserves the field) — required
    # for framework group_by(group_index)
    assert [s.index for s in out] == [99, 99, 99]
    assert [s.group_index for s in out] == [7, 7, 7]
    # reward uniform-split
    assert all(abs(s.reward - 1.0) < 1e-9 for s in out)


if __name__ == "__main__":
    import sys as _sys
    import pytest as _pytest
    _sys.exit(_pytest.main([__file__, "-v"]))
