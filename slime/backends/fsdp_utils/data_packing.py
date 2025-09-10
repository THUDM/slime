"""
Data packing utilities for FSDP backend to reduce padding overhead.
Supports packing variable-length sequences and dynamic micro batch size adjustment.
Integrates with slime's sequence length balancing utilities.
"""

import torch
from typing import List, Dict, Optional, Any
import numpy as np
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions


def pack_sequences(
    tokens: List[List[int]],
    loss_masks: List[List[int]],
    rewards: Optional[List[float]] = None,
    raw_rewards: Optional[List[Any]] = None,
    max_seq_len: Optional[int] = None,
    pack_efficiency_threshold: float = 0.8,
    use_dynamic_batch_size: bool = False,
    max_tokens_per_gpu: Optional[int] = None,
    num_packs: Optional[int] = None,
    partition_method: str = "greedy",
) -> Dict[str, Any]:
    """
    Pack multiple sequences into dense tensors with cumulative sequence lengths.
    
    Args:
        tokens: List of token sequences
        loss_masks: List of loss masks  
        rewards: Optional list of rewards
        raw_rewards: Optional list of raw rewards
        max_seq_len: Maximum sequence length (will truncate if exceeded)
        pack_efficiency_threshold: Minimum packing efficiency (ratio of actual tokens to total tokens)
        use_dynamic_batch_size: Enable dynamic micro batch size adjustment
        max_tokens_per_gpu: Maximum tokens per GPU when using dynamic batch size or static packing
        num_packs: If specified, create exactly this many packs using balanced partitioning
        partition_method: Method for creating partitions ('greedy' or 'balanced')
        
    Returns:
        Dictionary containing packed tensors and metadata
    """
    if use_dynamic_batch_size and max_tokens_per_gpu is None:
        raise ValueError("max_tokens_per_gpu must be specified when use_dynamic_batch_size is True")
    
    # Enforce max_seq_len if specified
    if max_seq_len is not None:
        for i, seq in enumerate(tokens):
            if len(seq) > max_seq_len:
                # Truncate sequences that exceed max_seq_len
                tokens[i] = seq[:max_seq_len]
                if i < len(loss_masks):
                    loss_masks[i] = loss_masks[i][:max_seq_len]
    
    seq_lengths = [len(seq) for seq in tokens]
    
    # Sort sequences by length for better packing efficiency
    sorted_indices = np.argsort(seq_lengths)[::-1]  # Descending order
    
    # Group sequences into packs
    packs = _create_packs(
        seq_lengths, 
        sorted_indices,
        use_dynamic_batch_size,
        max_tokens_per_gpu,
        pack_efficiency_threshold,
        num_packs,
        partition_method
    )
    
    packed_data = {
        "packs": [],
        "pack_metadata": [],
        "original_indices": sorted_indices.tolist(),
    }
    
    for pack_indices in packs:
        pack_data = _pack_single_batch(
            pack_indices,
            tokens,
            loss_masks,
            rewards,
            raw_rewards,
            sorted_indices,
        )
        packed_data["packs"].append(pack_data)
        packed_data["pack_metadata"].append({
            "num_sequences": len(pack_indices),
            "total_tokens": pack_data["total_tokens"],
            "efficiency": pack_data["efficiency"],
        })
    
    return packed_data


def _create_packs(
    seq_lengths: List[int],
    sorted_indices: np.ndarray,
    use_dynamic_batch_size: bool,
    max_tokens_per_gpu: Optional[int],
    pack_efficiency_threshold: float,
    num_packs: Optional[int] = None,
    partition_method: str = "greedy",
) -> List[List[int]]:
    """
    Create packs of sequences using either greedy heuristic or balanced partitioning.
    
    Args:
        seq_lengths: List of sequence lengths
        sorted_indices: Indices that sort sequences by length
        use_dynamic_batch_size: Whether to use dynamic batching
        max_tokens_per_gpu: Maximum tokens per GPU
        pack_efficiency_threshold: Minimum packing efficiency
        num_packs: If specified, create exactly this many packs
        partition_method: 'greedy' or 'balanced' partitioning
    """
    packs = []
    
    # Use balanced partitioning if num_packs is specified
    if num_packs is not None and partition_method == "balanced":
        # Create a mapping from original indices to sorted positions
        sorted_seq_lengths = [seq_lengths[idx] for idx in sorted_indices]
        
        # Use balanced partitioning on sorted sequences
        equal_size = len(sorted_indices) % num_packs == 0
        partitions = get_seqlen_balanced_partitions(
            sorted_seq_lengths,
            num_packs,
            equal_size
        )
        
        # partitions contains indices into the sorted sequence list
        # We keep these as-is since they map to positions in sorted_indices
        packs = partitions
    elif use_dynamic_batch_size:
        # Dynamic packing based on max_tokens_per_gpu
        current_pack = []
        current_tokens = 0
        
        for idx in range(len(sorted_indices)):
            seq_len = seq_lengths[sorted_indices[idx]]
            
            if current_tokens + seq_len <= max_tokens_per_gpu:
                current_pack.append(idx)
                current_tokens += seq_len
            else:
                if current_pack:
                    packs.append(current_pack)
                current_pack = [idx]
                current_tokens = seq_len
        
        if current_pack:
            packs.append(current_pack)
    else:
        # Static packing - try to pack sequences together efficiently
        pack_bins = []  # List of (current_max_len, indices, total_tokens)
        
        for idx in range(len(sorted_indices)):
            seq_len = seq_lengths[sorted_indices[idx]]
            best_bin = -1
            best_efficiency = 0
            
            # Try to find the best existing bin
            for bin_idx, (max_len, indices, total_tokens) in enumerate(pack_bins):
                new_max_len = max(max_len, seq_len)
                new_total_tokens = total_tokens + seq_len
                
                # Check token budget if specified
                if max_tokens_per_gpu is not None and new_total_tokens > max_tokens_per_gpu:
                    continue  # Would exceed budget
                
                new_efficiency = new_total_tokens / (new_max_len * (len(indices) + 1))
                
                if new_efficiency >= pack_efficiency_threshold and new_efficiency > best_efficiency:
                    best_bin = bin_idx
                    best_efficiency = new_efficiency
            
            if best_bin >= 0:
                # Add to existing bin
                max_len, indices, total_tokens = pack_bins[best_bin]
                indices.append(idx)
                pack_bins[best_bin] = (max(max_len, seq_len), indices, total_tokens + seq_len)
            else:
                # Create new bin
                pack_bins.append((seq_len, [idx], seq_len))
        
        packs = [indices for _, indices, _ in pack_bins]
    
    return packs


def _pack_single_batch(
    pack_indices: List[int],
    tokens: List[List[int]],
    loss_masks: List[List[int]],
    rewards: Optional[List[float]],
    raw_rewards: Optional[List[Any]],
    sorted_indices: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Pack a single batch of sequences into dense tensors.
    """
    # Get actual sequence indices
    # If sorted_indices is just a range (from seqlen_balancing), pack_indices are already actual indices
    if np.array_equal(sorted_indices, np.arange(len(sorted_indices))):
        actual_indices = pack_indices
    else:
        actual_indices = [sorted_indices[idx] for idx in pack_indices]
    
    # Get sequences for this pack
    pack_tokens = [tokens[idx] for idx in actual_indices]
    pack_loss_masks = [loss_masks[idx] for idx in actual_indices]
    seq_lengths = [len(seq) for seq in pack_tokens]
    
    # Calculate cumulative sequence lengths (for varlen attention)
    cu_seqlens = torch.zeros(len(pack_indices) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(seq_lengths, dtype=torch.int32).cumsum(0)
    
    # Flatten and concatenate all sequences
    flat_tokens = []
    flat_loss_masks = []
    
    for seq_tokens, seq_mask in zip(pack_tokens, pack_loss_masks):
        flat_tokens.extend(seq_tokens)
        # Robust mask alignment: ensure mask length matches token length
        if len(seq_mask) < len(seq_tokens):
            # Pad the tail with zeros to match token length
            pad = len(seq_tokens) - len(seq_mask)
            adjusted_mask = seq_mask + [0] * pad
        elif len(seq_mask) > len(seq_tokens):
            # Truncate to token length
            adjusted_mask = seq_mask[:len(seq_tokens)]
        else:
            adjusted_mask = seq_mask
        
        # Enforce LM convention: last token has no loss
        if adjusted_mask:
            adjusted_mask[-1] = 0
        
        flat_loss_masks.extend(adjusted_mask)
    
    # Convert to tensors
    packed_tokens = torch.tensor(flat_tokens, dtype=torch.long)
    packed_loss_masks = torch.tensor(flat_loss_masks, dtype=torch.int)
    
    # Calculate packing efficiency
    total_tokens = len(flat_tokens)
    max_seq_len = max(seq_lengths) if seq_lengths else 0
    padded_tokens = max_seq_len * len(pack_indices)
    efficiency = total_tokens / padded_tokens if padded_tokens > 0 else 1.0
    
    result = {
        "tokens": packed_tokens,
        "loss_masks": packed_loss_masks,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seq_len,
        "num_sequences": len(pack_indices),
        "total_tokens": total_tokens,
        "efficiency": efficiency,
        "sequence_indices": torch.tensor(actual_indices, dtype=torch.long),
    }
    
    # Add rewards if provided
    if rewards is not None:
        result["rewards"] = torch.tensor([rewards[idx] for idx in actual_indices], dtype=torch.float32)
    
    if raw_rewards is not None:
        result["raw_rewards"] = [raw_rewards[idx] for idx in actual_indices]
    
    return result


def unpack_sequences(
    packed_data: Dict[str, Any],
    original_order: bool = True,
) -> Dict[str, List]:
    """
    Unpack sequences from packed format back to list format.
    
    Args:
        packed_data: Packed data dictionary
        original_order: Whether to restore original sequence order
        
    Returns:
        Dictionary with unpacked sequences
    """
    items = []  # List of (orig_idx, seq_tokens, seq_loss_mask, reward, raw_reward)
    
    for pack in packed_data["packs"]:
        cu_seqlens = pack["cu_seqlens"]
        tokens = pack["tokens"]
        loss_masks = pack["loss_masks"]
        idxs = pack["sequence_indices"].tolist()
        
        # Unpack each sequence
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            
            seq_tokens = tokens[start:end].tolist()
            seq_loss_mask = loss_masks[start:end].tolist()
            
            reward = None
            if "rewards" in pack and i < len(pack["rewards"]):
                reward = pack["rewards"][i].item()
            
            raw_reward = None
            if "raw_rewards" in pack and i < len(pack["raw_rewards"]):
                raw_reward = pack["raw_rewards"][i]
            
            items.append((
                idxs[i],
                seq_tokens,
                seq_loss_mask,
                reward,
                raw_reward
            ))
    
    # Sort by original index if requested
    if original_order:
        items.sort(key=lambda x: x[0])
    
    # Build the unpacked dictionary
    unpacked = {
        "tokens": [],
        "loss_masks": [],
        "rewards": [],
        "raw_rewards": [],
    }
    
    for _, seq_tokens, seq_loss_mask, reward, raw_reward in items:
        unpacked["tokens"].append(seq_tokens)
        unpacked["loss_masks"].append(seq_loss_mask)
        if reward is not None:
            unpacked["rewards"].append(reward)
        if raw_reward is not None:
            unpacked["raw_rewards"].append(raw_reward)
    
    return unpacked


def compute_optimal_batch_size(
    seq_lengths: List[int],
    max_tokens_per_gpu: int,
    min_micro_batch_size: int = 1,
    max_micro_batch_size: int = 64,
    target_efficiency: float = 0.85,
    current_batch_size: Optional[int] = None,
    efficiency_history: Optional[List[float]] = None,
) -> int:
    """
    Compute optimal micro batch size based on sequence length distribution.
    
    Args:
        seq_lengths: List of sequence lengths in the current batch
        max_tokens_per_gpu: Maximum tokens per GPU
        min_micro_batch_size: Minimum micro batch size
        max_micro_batch_size: Maximum micro batch size
        target_efficiency: Target packing efficiency
        current_batch_size: Current micro batch size (for adjustment)
        efficiency_history: Recent efficiency history for adaptive adjustment
        
    Returns:
        Optimal micro batch size
    """
    if not seq_lengths:
        return min_micro_batch_size
    
    # Sort sequences by length
    sorted_lengths = sorted(seq_lengths, reverse=True)
    
    # Binary search for optimal batch size
    left, right = min_micro_batch_size, min(len(seq_lengths), max_micro_batch_size)
    best_batch_size = left
    
    while left <= right:
        mid = (left + right) // 2
        
        # Simulate packing with this batch size
        efficiency = _estimate_packing_efficiency(
            sorted_lengths, 
            mid,
            max_tokens_per_gpu
        )
        
        if efficiency >= target_efficiency:
            best_batch_size = mid
            left = mid + 1  # Try larger batch size
        else:
            right = mid - 1  # Try smaller batch size
    
    # Adaptive adjustment based on history
    if current_batch_size is not None and efficiency_history:
        history_window = 10
        recent_history = efficiency_history[-history_window:] if len(efficiency_history) > history_window else efficiency_history
        avg_efficiency = np.mean(recent_history)
        
        if avg_efficiency < target_efficiency * 0.9:
            # Reduce batch size if efficiency is consistently low
            best_batch_size = max(min_micro_batch_size, int(best_batch_size * 0.9))
        elif avg_efficiency > target_efficiency * 1.1:
            # Increase batch size if efficiency is consistently high
            best_batch_size = min(max_micro_batch_size, int(best_batch_size * 1.1))
    
    return best_batch_size


def _estimate_packing_efficiency(
    sorted_lengths: List[int],
    batch_size: int,
    max_tokens_per_gpu: int,
) -> float:
    """
    Estimate packing efficiency for a given batch size.
    """
    num_batches = (len(sorted_lengths) + batch_size - 1) // batch_size
    total_actual_tokens = sum(sorted_lengths)
    total_padded_tokens = 0
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, len(sorted_lengths))
        batch_lengths = sorted_lengths[batch_start:batch_end]
        
        if batch_lengths:
            max_len = max(batch_lengths)
            # Consider token budget constraint
            if max_tokens_per_gpu and max_len * len(batch_lengths) > max_tokens_per_gpu:
                # This batch would exceed budget, efficiency is poor
                return 0.0
            total_padded_tokens += max_len * len(batch_lengths)
    
    if total_padded_tokens == 0:
        return 1.0
    
    return total_actual_tokens / total_padded_tokens