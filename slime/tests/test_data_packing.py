# test_data_packing.py
import torch
import pytest

# at the very top of test_data_packing.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from slime.backends.fsdp_utils.data_packing import pack_sequences


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


def test_basic_packing():
    """Test basic packing functionality."""
    tokens = [
        _build_tokens(0, 5),
        _build_tokens(1, 3),
        _build_tokens(2, 7),
    ]
    masks = [_build_mask_same_length(t) for t in tokens]
    rewards = [0.1, 0.2, 0.3]
    raw_rewards = [{"id": i} for i in range(len(tokens))]
    
    packs = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        rewards=rewards,
        raw_rewards=raw_rewards,
    )
    
    assert len(packs) == 1  # Should create single pack
    pack = packs[0]
    
    # Check tensor types
    assert pack["tokens"].dtype == torch.long
    assert pack["loss_masks"].dtype == torch.int
    assert pack["cu_seqlens"].dtype == torch.int32
    assert pack["rewards"].dtype == torch.float32
    
    # Check cu_seqlens
    assert pack["cu_seqlens"].tolist() == [0, 5, 8, 15]
    assert pack["max_seqlen"] == 7
    
    # Check rewards preserved (with float tolerance)
    import numpy as np
    np.testing.assert_array_almost_equal(pack["rewards"].numpy(), rewards, decimal=5)
    assert pack["raw_rewards"] == raw_rewards


def test_max_tokens_per_gpu():
    """Test packing with token budget using balanced partitioning."""
    tokens = [
        _build_tokens(0, 10),
        _build_tokens(1, 10),
        _build_tokens(2, 10),
        _build_tokens(3, 10),
    ]
    masks = [_build_mask_same_length(t) for t in tokens]
    rewards = [0.1, 0.2, 0.3, 0.4]
    raw_rewards = [{"id": i} for i in range(len(tokens))]
    
    packs = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        rewards=rewards,
        raw_rewards=raw_rewards,
        max_tokens_per_gpu=20,
    )
    
    # With 40 total tokens and max 20, should create 2 balanced packs
    assert len(packs) == 2
    # Check that packs are balanced (each should have ~20 tokens)
    pack_sizes = [pack["cu_seqlens"][-1].item() for pack in packs]
    assert all(15 <= size <= 25 for size in pack_sizes)  # Allow some variance


def test_max_seq_len_truncation():
    """Test sequence truncation."""
    tokens = [
        _build_tokens(0, 10),
        _build_tokens(1, 4),
    ]
    masks = [_build_mask_same_length(t) for t in tokens]
    rewards = [0.1, 0.2]
    raw_rewards = [{"id": i} for i in range(len(tokens))]
    
    packs = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        rewards=rewards,
        raw_rewards=raw_rewards,
        max_seq_len=5,
    )
    
    pack = packs[0]
    assert pack["max_seqlen"] <= 5
    assert pack["cu_seqlens"].tolist() == [0, 5, 9]  # 5 + 4 tokens


def test_mask_alignment():
    """Test that masks are properly aligned."""
    tokens = [
        _build_tokens(0, 5),
        _build_tokens(1, 3),
    ]
    # Provide shorter masks
    masks = [[1, 1], [1]]
    rewards = [0.1, 0.2]
    raw_rewards = [{"id": i} for i in range(len(tokens))]
    
    packs = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        rewards=rewards,
        raw_rewards=raw_rewards,
    )
    
    pack = packs[0]
    # Masks should be padded and last token set to 0
    expected_masks = [1, 1, 0, 0, 0,  # First sequence
                      1, 0, 0]         # Second sequence
    assert pack["loss_masks"].tolist() == expected_masks


def test_empty_sequences():
    """Test with empty input."""
    packs = pack_sequences(
        tokens=[],
        loss_masks=[],
        rewards=[],
        raw_rewards=[],
    )
    assert packs == []


def test_balanced_partitioning():
    """Test that balanced partitioning creates well-balanced packs."""
    # Create sequences with varied lengths that would be poorly balanced with greedy
    tokens = [
        _build_tokens(0, 50),  # Large
        _build_tokens(1, 10),  # Small
        _build_tokens(2, 45),  # Large
        _build_tokens(3, 15),  # Small
        _build_tokens(4, 48),  # Large
        _build_tokens(5, 12),  # Small
    ]
    masks = [_build_mask_same_length(t) for t in tokens]
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    raw_rewards = [{"id": i} for i in range(len(tokens))]
    
    # Request 2 packs
    packs = pack_sequences(
        tokens=tokens,
        loss_masks=masks,
        rewards=rewards,
        raw_rewards=raw_rewards,
        num_packs=2,
    )
    
    assert len(packs) == 2
    
    # Check that packs are well balanced
    pack_sizes = [pack["cu_seqlens"][-1].item() for pack in packs]
    # Both packs should have roughly equal total tokens
    assert abs(pack_sizes[0] - pack_sizes[1]) <= 10  # Should be very close
    
    # Verify all sequences are included
    all_rewards = []
    for pack in packs:
        all_rewards.extend(pack["rewards"].tolist())
    assert len(all_rewards) == 6


def test_gather_log_probs_packed():
    """Test the simplified gather_log_probs_packed function."""
    from slime.backends.fsdp_utils.actor import gather_log_probs_packed
    
    # Create simple test data
    vocab_size = 100
    total_tokens = 10
    
    # Simulate packed sequences with cu_seqlens [0, 3, 7, 10]
    cu_seqlens = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
    input_ids = torch.randint(0, vocab_size, (total_tokens,))
    logits = torch.randn(total_tokens, vocab_size)
    
    log_probs = gather_log_probs_packed(logits, input_ids, cu_seqlens)
    
    # Should exclude first token of each sequence
    # So we get (3-1) + (4-1) + (3-1) = 2 + 3 + 2 = 7 log probs
    assert log_probs.shape == (7,)


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))