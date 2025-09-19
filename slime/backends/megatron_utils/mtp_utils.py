import torch

def generate_position_ids_for_mtp(args, tokens, packed_seq_params):
    """Generate position_ids for MTP training when needed.
    
    Args:
        args: Training arguments
        tokens: Input tokens tensor
        packed_seq_params: Packed sequence parameters (can be None)
        
    Returns:
        position_ids tensor or None if MTP training is not enabled
    """
    position_ids = None
    if getattr(args, "enable_mtp_training", False):
        seq_length = tokens.shape[1]
        if packed_seq_params is not None:
            # For packed sequences, position_ids should reset at each sequence boundary
            cu_seqlens = packed_seq_params.cu_seqlens_q
            position_ids = torch.zeros(seq_length, dtype=torch.long, device=tokens.device)
            for i in range(len(cu_seqlens) - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                seq_len = end - start
                position_ids[start:end] = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
            position_ids = position_ids.unsqueeze(0)
        else:
            # For non-packed sequences, standard position_ids
            position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
            position_ids = position_ids.unsqueeze(0).expand_as(tokens)
    return position_ids