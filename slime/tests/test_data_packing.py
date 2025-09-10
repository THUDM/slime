# test_data_packing.py
import math
import numpy as np
import pytest
import torch

# at the very top of test_data_packing.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from slime.backends.fsdp_utils.data_packing import pack_sequences, unpack_sequences, compute_optimal_batch_size

def _lens_from_cu(cu):
    cu = cu.cpu().tolist()
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]


def _total_tokens_from_pack(pack):
    return int(pack["cu_seqlens"][-1].item())


def _build_tokens(seq_id: int, length: int):
    # deterministic, unique content per sequence
    return list(range(seq_id * 1000, seq_id * 1000 + length))


def _build_mask_same_length(tokens):
    # typical LM convention: 1 on all but last token
    if len(tokens) == 0:
        return []
    m = [1] * len(tokens)
    m[-1] = 0
    return m


def _build_mask_len_minus_one(tokens):
    # one shorter than tokens (next-token loss style)
    if len(tokens) <= 1:
        return [0] * max(0, len(tokens) - 1)
    m = [1] * (len(tokens) - 1)
    # no explicit last 0 because shape is len-1
    return m


def _sum_over_packs(packed_data, key):
    return sum(pack[key] for pack in packed_data["packs"])


# ------------------------
# Sanity: cu_seqlens + shapes
# ------------------------

def test_cu_seqlens_and_shapes_consistent_dynamic():
    tokens = [
        _build_tokens(0, 5),
        _build_tokens(1, 3),
        _build_tokens(2, 7),
        _build_tokens(3, 1),
    ]
    masks = [_build_mask_same_length(t) for t in tokens]

    out = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=8  # enforce multiple packs
    )

    assert "packs" in out and len(out["packs"]) >= 1
    for pack in out["packs"]:
        toks = pack["tokens"]
        loss = pack["loss_masks"]
        cu = pack["cu_seqlens"]
        assert toks.dtype == torch.long
        assert cu.dtype in (torch.int32, torch.int64)  # author uses int32; accept int64 on CPU
        assert len(toks) == len(loss), "packed tokens and loss_masks lengths must match"
        assert cu[0].item() == 0
        assert cu[-1].item() == len(toks)
        # each segment end must align
        seg_lens = _lens_from_cu(cu)
        assert sum(seg_lens) == len(toks)
        # last element of each segment should not accrue loss if masks follow LM convention
        start = 0
        for L in seg_lens:
            if L > 0:
                assert loss[start + L - 1].item() == 0
            start += L

    # dynamic budget respected
    for pack in out["packs"]:
        assert _total_tokens_from_pack(pack) <= 8, "dynamic mode must respect token budget per pack"


# ------------------------
# Mask alignment corner cases (expected FAIL with current code if lengths==tokens)
# ------------------------

def test_mask_alignment_various_lengths_static():
    # Construct diverse mask length cases to ensure alignment == tokens length after packing
    t0 = _build_tokens(0, 5)   # mask == tokens
    m0 = [1, 1, 1, 1, 0]

    t1 = _build_tokens(1, 4)   # mask == tokens-1
    m1 = _build_mask_len_minus_one(t1)

    t2 = _build_tokens(2, 2)   # shorter than tokens by 1 (pathological but should be handled)
    m2 = [1]

    t3 = _build_tokens(3, 1)   # mask longer than tokens (very pathological; should be truncated)
    m3 = [1, 1]

    tokens = [t0, t1, t2, t3]
    masks = [m0, m1, m2, m3]

    out = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        pack_efficiency_threshold=0.0
    )

    for pack in out["packs"]:
        toks = pack["tokens"]
        loss = pack["loss_masks"]
        assert len(toks) == len(loss), (
            "Loss-mask alignment bug: packed loss mask length must equal packed tokens length."
        )

        # Validate per-sequence tails are zero (LM convention)
        cu = pack["cu_seqlens"]
        start = 0
        for L in _lens_from_cu(cu):
            if L > 0:
                assert loss[start + L - 1].item() == 0
            start += L


# ------------------------
# Original-order restoration (expected FAIL with current static binning implementation)
# ------------------------

def test_unpack_restores_original_order_static_bins():
    # Choose lengths that will interleave across bins under static packing
    lengths = [9, 4, 8, 3, 7, 2]
    tokens = [_build_tokens(i, L) for i, L in enumerate(lengths)]
    masks = [_build_mask_same_length(t) for t in tokens]

    packed = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        pack_efficiency_threshold=0.8
    )

    # Unpack and request original order
    unpacked = unpack_sequences(packed, original_order=True)

    # Compare exact content with original tokens (order and values)
    assert len(unpacked["tokens"]) == len(tokens)
    for i in range(len(tokens)):
        assert unpacked["tokens"][i] == tokens[i], (
            "Original order restoration failed for sequence index {}. "
            "Ensure unpack uses per-pack `sequence_indices` and sorts by them.".format(i)
        )


# ------------------------
# Efficiency accounting sanity
# ------------------------

def test_pack_efficiency_math_is_consistent():
    lengths = [30, 25, 10, 9, 5]  # scaled-down version of blog example
    tokens = [_build_tokens(i, L) for i, L in enumerate(lengths)]
    masks = [_build_mask_same_length(t) for t in tokens]

    out = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        pack_efficiency_threshold=0.85
    )

    # Each pack must satisfy: eff == total_tokens / (max_len * num_sequences)
    for pack in out["packs"]:
        tot = pack["total_tokens"]
        max_len = pack["max_seqlen"]
        nseq = pack["num_sequences"]
        denom = max_len * nseq if nseq > 0 else 1
        calc_eff = tot / denom
        assert math.isclose(calc_eff, pack["efficiency"], rel_tol=1e-6), "Per-pack efficiency mismatch"

    # Global check: sum of segment lengths equals sum of tokens
    sum_tokens = sum(len(t) for t in tokens)
    sum_flat = sum(pack["cu_seqlens"][-1].item() for pack in out["packs"])
    assert sum_tokens == sum_flat


# ------------------------
# Static-mode token budget (expected FAIL unless static path enforces cap)
# ------------------------

def test_static_mode_should_respect_token_budget_cap():
    lengths = [12, 11, 10, 9, 8, 7, 6]
    tokens = [_build_tokens(i, L) for i, L in enumerate(lengths)]
    masks = [_build_mask_same_length(t) for t in tokens]

    out = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        pack_efficiency_threshold=0.5,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=20  # desire cap even in static mode
    )

    # Expect all static packs to obey the token budget if provided
    for pack in out["packs"]:
        assert _total_tokens_from_pack(pack) <= 20, (
            "Static mode should also guard by max_tokens_per_gpu (or a max_tokens_per_pack)."
        )


# ------------------------
# max_seq_len enforcement (expected FAIL with current code)
# ------------------------

def test_max_seq_len_is_enforced_or_truncated():
    # One sequence exceeds the configured max_seq_len
    tokens = [
        _build_tokens(0, 6),  # exceeds
        _build_tokens(1, 4),
    ]
    masks = [_build_mask_same_length(t) for t in tokens]

    packed = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        max_seq_len=5,
        pack_efficiency_threshold=0.0
    )

    # All packs must have max_seqlen <= configured max_seq_len if enforcement is present.
    # With current code this will FAIL because nothing enforces max_seq_len.
    for pack in packed["packs"]:
        assert pack["max_seqlen"] <= 5, (
            "max_seq_len is not enforced. Either truncate inputs or raise early."
        )


# ------------------------
# Zero-length sequence tolerance (should PASS)
# ------------------------

def test_zero_length_sequences_do_not_crash_and_mark_zero_segments():
    tokens = [
        _build_tokens(0, 0),
        _build_tokens(1, 3),
        _build_tokens(2, 0),
    ]
    masks = [_build_mask_same_length(t) for t in tokens]

    out = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        pack_efficiency_threshold=0.0
    )

    # Ensure cu_seqlens includes zero-length segments and total tokens equals sum of lengths
    total = sum(len(t) for t in tokens)
    sum_flat = sum(pack["cu_seqlens"][-1].item() for pack in out["packs"])
    assert total == sum_flat

    for pack in out["packs"]:
        lens = _lens_from_cu(pack["cu_seqlens"])
        # All lengths must be >= 0
        assert all(L >= 0 for L in lens)


# ------------------------
# Rewards / raw_rewards roundtrip via unpack (exposes order issues)
# ------------------------

def test_rewards_roundtrip_after_unpack():
    lengths = [5, 3, 7, 1]
    tokens = [_build_tokens(i, L) for i, L in enumerate(lengths)]
    masks = [_build_mask_same_length(t) for t in tokens]
    rewards = [0.1, 0.2, 0.3, 0.4]
    raw_rewards = [{"foo": i} for i in range(len(tokens))]

    packed = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        rewards=rewards,
        raw_rewards=raw_rewards,
        pack_efficiency_threshold=0.8
    )

    unpacked = unpack_sequences(packed, original_order=True)
    
    # Check that rewards match with appropriate floating point tolerance
    assert len(unpacked["rewards"]) == len(rewards)
    for i in range(len(rewards)):
        assert math.isclose(unpacked["rewards"][i], rewards[i], rel_tol=1e-6), (
            f"Rewards mismatch at index {i}"
        )
    assert unpacked["raw_rewards"] == raw_rewards


# ------------------------
# Test compute_optimal_batch_size function
# ------------------------

def test_compute_optimal_batch_size():
    seq_lengths = [512, 480, 350, 256, 128, 64, 32]
    
    # Test basic functionality
    batch_size = compute_optimal_batch_size(
        seq_lengths=seq_lengths,
        max_tokens_per_gpu=2048,
        min_micro_batch_size=1,
        max_micro_batch_size=8,
        target_efficiency=0.85
    )
    
    assert 1 <= batch_size <= 8, "Batch size out of bounds"
    
    # Test with history for adaptive adjustment
    efficiency_history = [0.75, 0.78, 0.80, 0.82, 0.83]
    batch_size_with_history = compute_optimal_batch_size(
        seq_lengths=seq_lengths,
        max_tokens_per_gpu=2048,
        current_batch_size=4,
        efficiency_history=efficiency_history
    )
    
    # Should reduce batch size due to low efficiency
    assert batch_size_with_history <= 4
    
    # Test empty sequences
    empty_batch_size = compute_optimal_batch_size(
        seq_lengths=[],
        max_tokens_per_gpu=2048
    )
    assert empty_batch_size == 1  # Should return minimum

def test_balanced_partitioning():
    # Test the new balanced partitioning feature
    lengths = [100, 200, 150, 300, 250, 120, 180, 220, 90, 110]
    tokens = [_build_tokens(i, L) for i, L in enumerate(lengths)]
    masks = [_build_mask_same_length(t) for t in tokens]
    
    # Test with specified number of packs using balanced partitioning
    packed = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        num_packs=3,
        partition_method="balanced"
    )
    
    # Should create exactly 3 packs
    assert len(packed["packs"]) == 3
    
    # Check that all sequences are accounted for
    total_sequences = sum(pack["num_sequences"] for pack in packed["packs"])
    assert total_sequences == len(tokens)
    
    # Verify load balancing - calculate total tokens per pack
    pack_loads = []
    for pack in packed["packs"]:
        pack_loads.append(pack["total_tokens"])
    
    # Balanced partitioning should minimize variance in pack loads
    max_load = max(pack_loads)
    min_load = min(pack_loads)
    avg_load = sum(pack_loads) / len(pack_loads)
    
    # Check that the load imbalance is reasonable (within 20% of average)
    assert (max_load - min_load) / avg_load < 0.4
    
    # Verify that unpacking restores all sequences correctly
    unpacked = unpack_sequences(packed, original_order=True)
    assert len(unpacked["tokens"]) == len(tokens)
    
    # Test with equal-size constraint (when divisible)
    tokens_equal = tokens[:9]  # 9 sequences, divisible by 3
    masks_equal = masks[:9]
    
    packed_equal = pack_sequences(
        tokens=tokens_equal,
        loss_masks=masks_equal,
        num_packs=3,
        partition_method="balanced"
    )
    
    # Each pack should have exactly 3 sequences
    for pack in packed_equal["packs"]:
        assert pack["num_sequences"] == 3

    
if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))