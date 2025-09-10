# test_data_packing.py
import math
import numpy as np
import pytest
import torch

# at the very top of test_data_packing.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from slime.backends.fsdp_utils.data_packing import pack_sequences, compute_optimal_batch_size


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


def test_cu_seqlens_and_shapes_consistent():
    """Test that cu_seqlens are consistent with packed tensor shapes."""
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
        max_tokens_per_gpu=8  # enforce multiple packs
    )

    assert "packs" in out and len(out["packs"]) >= 1
    
    for pack in out["packs"]:
        toks = pack["tokens"]
        loss = pack["loss_masks"]
        cu = pack["cu_seqlens"]
        
        assert toks.dtype == torch.long
        assert cu.dtype in (torch.int32, torch.int64)
        assert len(toks) >= len(loss), "tokens should be >= loss_masks due to padding"
        assert cu[0].item() == 0
        # cu_seqlens includes padding
        assert cu[-1].item() == len(toks)
        
        # Check that each segment's last token has no loss
        seg_lens = _lens_from_cu(cu[:pack["num_sequences"]+1])
        start = 0
        for L in seg_lens:
            if L > 0:
                assert loss[start + L - 1].item() == 0
            start += L

    # Check token budget is respected
    for pack in out["packs"]:
        # Only check actual tokens (without padding)
        assert pack["total_tokens"] <= 8


def test_mask_alignment_various_lengths():
    """Test that masks are properly aligned with tokens."""
    t0 = _build_tokens(0, 5)
    m0 = [1, 1, 1, 1, 0]

    t1 = _build_tokens(1, 4)
    m1 = [1, 1, 1]  # shorter than tokens

    t2 = _build_tokens(2, 2)
    m2 = [1]  # much shorter

    t3 = _build_tokens(3, 3)
    m3 = [1, 1, 1, 1, 1]  # longer than tokens

    tokens = [t0, t1, t2, t3]
    masks = [m0, m1, m2, m3]

    out = pack_sequences(tokens=tokens, loss_masks=masks)

    for pack in out["packs"]:
        toks = pack["tokens"]
        loss = pack["loss_masks"]
        # After alignment and padding, lengths should match
        assert len(toks) == len(loss)

        # Check last token of each sequence has no loss
        cu = pack["cu_seqlens"]
        start = 0
        for i in range(pack["num_sequences"]):
            end = cu[i + 1].item()
            if end > start:
                # Find actual sequence end (before padding)
                seq_end = min(end, cu[i].item() + len(tokens[i]))
                if seq_end > start:
                    assert loss[seq_end - 1].item() == 0
            start = end


def test_max_seq_len_truncation():
    """Test that sequences are truncated to max_seq_len."""
    tokens = [
        _build_tokens(0, 10),  # will be truncated
        _build_tokens(1, 4),
    ]
    masks = [_build_mask_same_length(t) for t in tokens]

    packed = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        max_seq_len=5
    )

    for pack in packed["packs"]:
        assert pack["max_seqlen"] <= 5


def test_zero_length_sequences():
    """Test that zero-length sequences don't crash."""
    tokens = [
        _build_tokens(0, 0),
        _build_tokens(1, 3),
        _build_tokens(2, 0),
    ]
    masks = [_build_mask_same_length(t) for t in tokens]

    out = pack_sequences(tokens=tokens, loss_masks=masks)

    # Should handle empty sequences gracefully
    total = sum(len(t) for t in tokens)
    sum_flat = sum(pack["total_tokens"] for pack in out["packs"])
    assert total == sum_flat


def test_rewards_preserved():
    """Test that rewards are correctly preserved through packing."""
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
    )

    # Check rewards are preserved
    all_rewards = []
    all_raw_rewards = []
    for pack in packed["packs"]:
        if "rewards" in pack:
            all_rewards.extend(pack["rewards"].tolist())
        if "raw_rewards" in pack:
            all_raw_rewards.extend(pack["raw_rewards"])
    
    assert len(all_rewards) == len(rewards)
    assert len(all_raw_rewards) == len(raw_rewards)


def test_padding_to_multiple():
    """Test padding to multiple of 128."""
    tokens = [
        _build_tokens(0, 50),
        _build_tokens(1, 30),
    ]
    masks = [_build_mask_same_length(t) for t in tokens]

    packed = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        pad_to_multiple=128
    )

    for pack in packed["packs"]:
        # Check that token tensor length is multiple of 128
        assert len(pack["tokens"]) % 128 == 0
        # But total_tokens should reflect actual tokens
        assert pack["total_tokens"] == sum(len(t) for t in tokens)


def test_compute_optimal_batch_size():
    """Test optimal batch size computation."""
    seq_lengths = [512, 480, 350, 256, 128, 64, 32]
    
    # Test basic functionality
    batch_size = compute_optimal_batch_size(
        seq_lengths=seq_lengths,
        max_tokens_per_gpu=2048,
        min_micro_batch_size=1,
        max_micro_batch_size=8,
        target_efficiency=0.85
    )
    
    assert 1 <= batch_size <= 8
    
    # Test with different parameters
    batch_size_2 = compute_optimal_batch_size(
        seq_lengths=seq_lengths,
        max_tokens_per_gpu=1024,  # Smaller budget
        min_micro_batch_size=2,
        max_micro_batch_size=4,
        target_efficiency=0.9
    )
    
    # Should return smaller batch size due to tighter constraints
    assert 2 <= batch_size_2 <= 4
    
    # Test empty sequences
    empty_batch_size = compute_optimal_batch_size(
        seq_lengths=[],
        max_tokens_per_gpu=2048
    )
    assert empty_batch_size == 1  # Should return minimum


def test_token_budget_enforcement():
    """Test that max_tokens_per_gpu is enforced."""
    lengths = [12, 11, 10, 9, 8, 7, 6]
    tokens = [_build_tokens(i, L) for i, L in enumerate(lengths)]
    masks = [_build_mask_same_length(t) for t in tokens]

    out = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        max_tokens_per_gpu=20
    )

    # All packs should respect token budget
    for pack in out["packs"]:
        assert pack["total_tokens"] <= 20


def test_efficiency_calculation():
    """Test that efficiency is calculated correctly."""
    lengths = [10, 10, 5, 5]  # Will pack well together
    tokens = [_build_tokens(i, L) for i, L in enumerate(lengths)]
    masks = [_build_mask_same_length(t) for t in tokens]

    out = pack_sequences(tokens=tokens, loss_masks=masks)

    for pack in out["packs"]:
        # Efficiency = actual_tokens / (max_seq_len * num_sequences)
        expected_eff = pack["total_tokens"] / (pack["max_seqlen"] * pack["num_sequences"])
        assert math.isclose(pack["efficiency"], expected_eff, rel_tol=1e-6)


def test_balanced_partitioning():
    """Test balanced partitioning using seqlen_balancing."""
    lengths = [100, 200, 150, 300, 250, 120]
    tokens = [_build_tokens(i, L) for i, L in enumerate(lengths)]
    masks = [_build_mask_same_length(t) for t in tokens]
    
    # Test with specified number of packs
    packed = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        num_packs=2  # Create exactly 2 balanced packs
    )
    
    # Should create exactly 2 packs
    assert len(packed["packs"]) == 2
    
    # Check that all sequences are accounted for
    total_sequences = sum(pack["num_sequences"] for pack in packed["packs"])
    assert total_sequences == len(tokens)
    
    # Verify load balancing - calculate total tokens per pack
    pack_loads = [pack["total_tokens"] for pack in packed["packs"]]
    
    # Balanced partitioning should minimize variance
    max_load = max(pack_loads)
    min_load = min(pack_loads)
    avg_load = sum(pack_loads) / len(pack_loads)
    
    # Check that the load imbalance is reasonable (within 20% of average)
    assert (max_load - min_load) / avg_load < 0.4

    
if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))