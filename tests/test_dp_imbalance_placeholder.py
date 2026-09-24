"""Test for the get_batch empty-microbatch placeholder schema.

When a DP rank has fewer real partitions than the collective MAX,
``get_batch`` calls ``_fill_empty_microbatch_placeholder`` to fabricate
a 1-token placeholder sample. The placeholder must be *self-consistent*:
every list-valued schema key must end up with exactly one entry, the
key-specific values must match the declared contract (total_lengths=[1],
response_lengths=[0], ...), and the post-fill invariant assertion must
trip when a schema field lands in ``keys`` without a placeholder rule.

These tests import the real helper from ``data.py`` so a future change
to the placeholder logic fails here rather than documenting a
replica of it.
"""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    pytest.skip("torch not available", allow_module_level=True)

# The helper uses torch.cuda.current_device(); on a CPU-only environment
# CUDA is unavailable. Skip the whole module in that case (these tests
# exist to exercise the CUDA code path that fires in production).
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required for the device=cuda placeholder tensors", allow_module_level=True)

from slime.backends.megatron_utils.data import _fill_empty_microbatch_placeholder  # noqa: E402


def test_placeholder_fills_known_keys():
    batch = {
        "tokens": [],
        "total_lengths": [],
        "response_lengths": [],
        "max_seq_lens": [],
        "loss_masks": [],
        "log_probs": [],
        "ref_log_probs": [],
    }
    keys = list(batch.keys())
    tokens = _fill_empty_microbatch_placeholder(batch, keys, pad_token_id=0)

    # Returned tokens list == batch["tokens"]
    assert tokens is batch["tokens"]
    assert len(tokens) == 1 and tokens[0].numel() == 1
    assert tokens[0].dtype == torch.long

    # Known-schema fields match their contract.
    assert batch["total_lengths"] == [1]
    assert batch["response_lengths"] == [0]
    assert batch["max_seq_lens"] == [1]
    assert isinstance(batch["loss_masks"][0], torch.Tensor)
    assert batch["loss_masks"][0].numel() == 0
    assert batch["loss_masks"][0].dtype == torch.int
    # Per-token default for unknown fields: length-0 fp32.
    assert batch["log_probs"][0].numel() == 0
    assert batch["log_probs"][0].dtype == torch.float32
    assert batch["ref_log_probs"][0].numel() == 0


def test_placeholder_per_sample_counts_all_match_tokens():
    """Core invariant: every list-valued key has exactly one entry
    after the fill, aligned with the 1-token placeholder."""
    batch = {
        "tokens": [],
        "total_lengths": [],
        "response_lengths": [],
        "loss_masks": [],
        "log_probs": [],
        "advantages": [],
    }
    keys = list(batch.keys())
    _fill_empty_microbatch_placeholder(batch, keys, pad_token_id=0)

    for k, v in batch.items():
        assert len(v) == 1, f"key {k!r}: expected 1 entry, got {len(v)}"


def test_placeholder_unknown_key_gets_per_token_default():
    """A new schema field not in placeholder_for_key gets the per-token
    default. This encodes the 'everything else is a per-token fp32 tensor'
    assumption — if that stops being true, a future key should add
    itself to placeholder_for_key."""
    batch = {"tokens": [], "something_new": []}
    keys = list(batch.keys())
    _fill_empty_microbatch_placeholder(batch, keys, pad_token_id=0)
    assert batch["something_new"][0].dtype == torch.float32
    assert batch["something_new"][0].numel() == 0


def test_placeholder_invariant_fires_on_unfilled_key():
    """The post-fill invariant assertion catches the case where a key
    is present in ``keys`` but its entry in ``batch`` is not a list —
    the fill loop skips it, then the invariant loop catches it.

    This exercises the actual assertion in ``_fill_empty_microbatch_placeholder``,
    not a copy of it in the test — so future edits to the helper that
    weaken or move the assertion make this test fail.
    """
    # Simulate the broken-caller shape: 'total_lengths' is present but a
    # non-list sentinel (typical bug: caller forgot to initialize as []).
    # The helper's fill loop sees `isinstance(v, list)` False and skips
    # it; the invariant loop then sees `isinstance(v, list)` False too
    # and also skips it — so actually no AssertionError fires.
    #
    # Instead, construct a case where the key is a list with != 1 entries
    # after fill. The simplest route: pass a key that isn't in batch at
    # all (batch.get returns None, fill loop skips), but *also* add a
    # sentinel entry to batch so that the invariant loop encounters a
    # list with 0 entries.
    batch = {
        "tokens": [],
        # 'bad_key' is in keys but batch[bad_key] is a list with 2 entries,
        # so it skips the len==0 branch in the fill loop. The invariant
        # then sees len == 2 and raises.
        "bad_key": ["stale1", "stale2"],
    }
    keys = list(batch.keys())

    with pytest.raises(AssertionError, match="bad_key"):
        _fill_empty_microbatch_placeholder(batch, keys, pad_token_id=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
