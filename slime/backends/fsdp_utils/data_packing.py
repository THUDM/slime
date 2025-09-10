"""Data packing utilities for FSDP backend to reduce padding overhead."""
import torch
from typing import List, Dict, Optional, Any
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions

def pack_sequences(
    tokens: List[List[int]],
    loss_masks: List[List[int]],
    rewards: Optional[List[float]] = None,
    raw_rewards: Optional[List[Any]] = None,
    max_seq_len: Optional[int] = None,
    max_tokens_per_gpu: Optional[int] = None,
    num_packs: Optional[int] = None,
    pad_to_multiple: int = 128,
) -> Dict[str, Any]:
    """
    Pack sequences into dense tensors with cumulative sequence lengths.
    
    Args:
        tokens: List of token sequences
        loss_masks: List of loss masks  
        rewards: Optional list of rewards
        raw_rewards: Optional list of raw rewards
        max_seq_len: Maximum sequence length (will truncate if exceeded)
        max_tokens_per_gpu: Maximum tokens per GPU (creates packs respecting this limit)
        num_packs: If specified, create exactly this many balanced packs
        pad_to_multiple: Pad total tokens to multiple of this (for memory efficiency)
        
    Returns:
        Dictionary containing packed tensors and metadata
    """
    if not tokens:
        return {"packs": []}
    
    # Truncate sequences exceeding max_seq_len
    if max_seq_len:
        tokens = [t[:max_seq_len] for t in tokens]
        loss_masks = [m[:max_seq_len] for m in loss_masks]
    
    seq_lengths = [len(t) for t in tokens]
    
    # Determine packing strategy
    if num_packs:
        # Use balanced partitioning for specified number of packs
        equal_size = len(tokens) % num_packs == 0
        packs = get_seqlen_balanced_partitions(seq_lengths, num_packs, equal_size)
    elif max_tokens_per_gpu:
        # Create packs respecting token budget
        packs = _create_token_limited_packs(seq_lengths, max_tokens_per_gpu)
    else:
        # Single pack with all sequences
        packs = [[i for i in range(len(tokens))]]
    
    packed_data = {"packs": []}
    
    for pack_indices in packs:
        pack = _pack_single_batch(
            pack_indices, tokens, loss_masks, 
            rewards, raw_rewards, pad_to_multiple
        )
        packed_data["packs"].append(pack)
    
    return packed_data


def _create_token_limited_packs(
    seq_lengths: List[int], 
    max_tokens: int
) -> List[List[int]]:
    """Create packs respecting max token limit using greedy approach."""
    # Sort by length descending for better packing
    sorted_indices = sorted(range(len(seq_lengths)), key=lambda i: seq_lengths[i], reverse=True)
    
    packs = []
    current_pack = []
    current_tokens = 0
    
    for idx in sorted_indices:
        seq_len = seq_lengths[idx]
        if current_tokens + seq_len > max_tokens and current_pack:
            packs.append(current_pack)
            current_pack = []
            current_tokens = 0
        current_pack.append(idx)
        current_tokens += seq_len
    
    if current_pack:
        packs.append(current_pack)
    
    return packs


def _pack_single_batch(
    indices: List[int],
    tokens: List[List[int]],
    loss_masks: List[List[int]],
    rewards: Optional[List[float]],
    raw_rewards: Optional[List[Any]],
    pad_to_multiple: int = 128,
) -> Dict[str, torch.Tensor]:
    """Pack a single batch into dense tensors."""
    # Get sequences for this pack
    pack_tokens = [tokens[i] for i in indices]
    pack_masks = [loss_masks[i] for i in indices]
    
    # Build cumulative sequence lengths
    cu_seqlens = [0]
    for seq in pack_tokens:
        cu_seqlens.append(cu_seqlens[-1] + len(seq))
    
    # Concatenate all sequences
    flat_tokens = []
    flat_masks = []
    
    for seq_tokens, seq_mask in zip(pack_tokens, pack_masks):
        flat_tokens.extend(seq_tokens)
        
        # Align mask length with tokens
        mask = seq_mask[:len(seq_tokens)]  # Truncate if longer
        mask.extend([0] * (len(seq_tokens) - len(mask)))  # Pad if shorter
        
        # LM convention: last token has no loss
        if mask:
            mask[-1] = 0
        
        flat_masks.extend(mask)
    
    # Pad to multiple for memory efficiency
    total_tokens = len(flat_tokens)
    if pad_to_multiple > 1:
        pad = (pad_to_multiple - total_tokens % pad_to_multiple) % pad_to_multiple
        if pad:
            flat_tokens.extend([0] * pad)  # Use 0 as pad token
            flat_masks.extend([0] * pad)
            # Don't add extra cu_seqlens entry - padding is just extra tokens at the end
    
    # Convert to tensors
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)
    max_seqlen = max(cu_seqlens[i+1] - cu_seqlens[i] for i in range(len(indices)))
    
    result = {
        "tokens": torch.tensor(flat_tokens, dtype=torch.long),
        "loss_masks": torch.tensor(flat_masks, dtype=torch.int),
        "cu_seqlens": cu_seqlens_tensor,
        "max_seqlen": max_seqlen,
        "num_sequences": len(indices),
        "total_tokens": total_tokens,  # Actual tokens (without padding)
        "efficiency": total_tokens / (max_seqlen * len(indices)) if indices else 1.0,
    }
    
    # Add rewards if provided
    if rewards:
        result["rewards"] = torch.tensor([rewards[i] for i in indices], dtype=torch.float32)
    if raw_rewards:
        result["raw_rewards"] = [raw_rewards[i] for i in indices]
    
    return result


def compute_optimal_batch_size(
    seq_lengths: List[int],
    max_tokens_per_gpu: int,
    min_micro_batch_size: int = 1,
    max_micro_batch_size: int = 64,
    target_efficiency: float = 0.85,
) -> int:
    """Compute optimal micro batch size based on sequence lengths."""
    if not seq_lengths:
        return min_micro_batch_size
    
    sorted_lengths = sorted(seq_lengths, reverse=True)
    left, right = min_micro_batch_size, min(len(seq_lengths), max_micro_batch_size)
    best_batch_size = left
    
    while left <= right:
        mid = (left + right) // 2
        total_actual = sum(sorted_lengths)
        total_padded = 0
        
        for i in range(0, len(sorted_lengths), mid):
            batch = sorted_lengths[i:i+mid]
            if batch:
                max_len = max(batch)
                if max_tokens_per_gpu and max_len * len(batch) > max_tokens_per_gpu:
                    total_padded = float('inf')  # Exceeds budget
                    break
                total_padded += max_len * len(batch)
        
        efficiency = total_actual / total_padded if total_padded < float('inf') else 0.0
        
        if efficiency >= target_efficiency:
            best_batch_size = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return best_batch_size
