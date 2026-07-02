"""mRoPE helpers for text-only Megatron training batches."""

import torch


def _build_local_thd_cp_position_ids(batch):
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    if cp_size == 1:
        return None

    device = batch["tokens"].device
    parts = []
    for tokens in batch["unconcat_tokens"]:
        token_len = int(tokens.size(0))
        chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
        for start in (
            cp_rank * chunk_size,
            (2 * cp_size - cp_rank - 1) * chunk_size,
        ):
            position = torch.arange(start, start + chunk_size, device=device, dtype=torch.long)
            parts.append(torch.where(position < token_len, position, torch.zeros_like(position)))

    if not parts:
        return None

    position = torch.cat(parts)
    target_len = int(batch["tokens"].numel())
    if position.numel() > target_len:
        raise ValueError(f"local mRoPE position length {position.numel()} exceeds token length {target_len}")
    if position.numel() < target_len:
        position = torch.cat(
            [
                position,
                torch.zeros(target_len - position.numel(), device=device, dtype=position.dtype),
            ]
        )
    return position.view(1, 1, target_len).expand(3, 1, target_len).contiguous()


def build_mrope_position_ids(batch, *, local_thd_cp: bool = False):
    """Return position ids of shape [3, 1, max_seqlen] for text-only mRoPE.

    For THD, mirror the packed sequence max length so Megatron can slice per
    document. For BSHD, packed sequence params are intentionally absent, so
    use the padded sequence length from the token tensor. Text-only data uses
    identical temporal/height/width axes.
    """
    if local_thd_cp:
        position_ids = _build_local_thd_cp_position_ids(batch)
        if position_ids is not None:
            return position_ids

    packed_seq_params = batch["packed_seq_params"]
    if packed_seq_params is None:
        max_seqlen = int(batch["tokens"].shape[-1])
    else:
        max_seqlen = int(packed_seq_params.max_seqlen_q)

    position = torch.arange(max_seqlen, device=batch["tokens"].device, dtype=torch.long)
    return position.view(1, 1, max_seqlen).expand(3, 1, max_seqlen).contiguous()
