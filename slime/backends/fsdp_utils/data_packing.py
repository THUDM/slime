"""Data packing utilities for FSDP backend to reduce padding overhead."""
import math
import torch
from typing import List, Dict, Optional
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions


def pack_sequences(
    tokens: List[List[int]],
    loss_masks: List[List[int]],
    rewards: List[float],
    raw_rewards: List,
    max_seq_len: Optional[int] = None,
    max_tokens_per_gpu: Optional[int] = None,
    num_packs: Optional[int] = None,
) -> List[Dict]:
    """
    Pack sequences into dense batches with cumulative sequence lengths.
    
    Args:
        tokens: List of token sequences
        loss_masks: List of loss masks
        rewards: List of rewards per sequence
        raw_rewards: List of raw rewards per sequence
        max_seq_len: Maximum sequence length (will truncate if exceeded)
        max_tokens_per_gpu: Maximum tokens per GPU pack
        num_packs: Explicit number of packs to create
        
    Returns:
        List of packed batches with tokens, masks, cu_seqlens, and rewards
    """
    if not tokens:
        return []
    
    # Truncate sequences if needed
    if max_seq_len:
        tokens = [t[:max_seq_len] for t in tokens]
        loss_masks = [m[:max_seq_len] for m in loss_masks]
    
    seq_lengths = [len(t) for t in tokens]
    
    # Determine number of packs and use balanced partitioning
    if num_packs:
        k_partitions = num_packs
    elif max_tokens_per_gpu:
        total_tokens = sum(seq_lengths)
        k_partitions = max(1, math.ceil(total_tokens / max_tokens_per_gpu))
    else:
        k_partitions = 1
    
    # Use balanced partitioning for optimal load distribution
    partitions = get_seqlen_balanced_partitions(
        seq_lengths, 
        k_partitions=k_partitions,
        equal_size=False  # Allow variable sizes for better balance
    )
    
    # Pack each partition
    result = []
    for indices in partitions:
        # Build cumulative sequence lengths
        cu_seqlens = [0]
        flat_tokens = []
        flat_masks = []
        
        for i in indices:
            seq_tokens = tokens[i]
            seq_mask = loss_masks[i]
            
            flat_tokens.extend(seq_tokens)
            
            # Align mask with tokens and set last token to 0 (LM convention)
            mask = seq_mask[:len(seq_tokens)]
            mask.extend([0] * (len(seq_tokens) - len(mask)))
            if mask:
                mask[-1] = 0
            flat_masks.extend(mask)
            
            cu_seqlens.append(cu_seqlens[-1] + len(seq_tokens))
        
        result.append({
            "tokens": torch.tensor(flat_tokens, dtype=torch.long),
            "loss_masks": torch.tensor(flat_masks, dtype=torch.int),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "max_seqlen": max(cu_seqlens[i+1] - cu_seqlens[i] for i in range(len(indices))),
            "rewards": torch.tensor([rewards[i] for i in indices], dtype=torch.float32),
            "raw_rewards": [raw_rewards[i] for i in indices],
        })
    
    return result