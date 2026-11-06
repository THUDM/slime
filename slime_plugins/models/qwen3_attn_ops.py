"""Trainable wrappers around SGLang Qwen3 attention kernels.

Forward stays on the SGLang kernels used by rollout (fused QK-Norm+RoPE, FA4).
Backward is a separate, autograd-visible implementation so QKV still receives
gradients — the same split GLM-5 uses for FlashMLA + TileLang bwd.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_ROPE_CACHE: dict = {}


def _get_rope_cache(
    device: torch.device,
    rotary_dim: int,
    rotary_base: float,
    needed_positions: int,
) -> torch.Tensor:
    key = (device.type, device.index, rotary_dim, float(rotary_base))
    cache = _ROPE_CACHE.get(key)
    if cache is not None and cache.shape[0] >= needed_positions:
        return cache
    inv_freq = 1.0 / (
        rotary_base
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    positions = torch.arange(needed_positions, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    _ROPE_CACHE[key] = cache
    return cache


def _neox_rope_forward(
    value: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace

    output = torch.empty_strided(
        value.size(), value.stride(), dtype=value.dtype, device=value.device
    )
    output.copy_(value)
    dummy_k = torch.empty(
        (value.shape[0], 1, value.shape[-1]),
        dtype=value.dtype,
        device=value.device,
    )
    apply_rope_with_cos_sin_cache_inplace(
        output,
        dummy_k,
        cos_sin_cache,
        positions,
        is_neox=True,
    )
    return output


def _neox_rope_backward(
    grad_output: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    half = grad_output.shape[-1] // 2
    broadcast_shape = (positions.numel(),) + (1,) * (grad_output.ndim - 2) + (half,)
    cos = cos_sin_cache[positions, :half].view(broadcast_shape).to(grad_output.dtype)
    sin = cos_sin_cache[positions, half:].view(broadcast_shape).to(grad_output.dtype)
    grad_first = grad_output[..., :half]
    grad_second = grad_output[..., half:]
    return torch.cat(
        (
            grad_first * cos + grad_second * sin,
            grad_second * cos - grad_first * sin,
        ),
        dim=-1,
    )


class _NeoXRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, cos_sin_cache, positions):
        ctx.save_for_backward(cos_sin_cache, positions)
        return _neox_rope_forward(value, cos_sin_cache, positions)

    @staticmethod
    def backward(ctx, grad_output):
        cos_sin_cache, positions = ctx.saved_tensors
        return _neox_rope_backward(grad_output, cos_sin_cache, positions), None, None


def neox_rope(
    value: torch.Tensor,
    positions: torch.Tensor,
    rotary_base: float,
) -> torch.Tensor:
    cache = _get_rope_cache(
        value.device,
        value.shape[-1],
        rotary_base,
        int(positions.max().item()) + 1,
    )
    if torch.is_grad_enabled() and value.requires_grad:
        return _NeoXRoPE.apply(value, cache, positions)
    return _neox_rope_forward(value, cache, positions)


def _rmsnorm_backward(
    grad_output: torch.Tensor,
    input_: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = input_.float()
    grad = grad_output.float()
    w = weight.float()
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    xhat = x * rstd
    reduce_dims = tuple(range(grad.ndim - 1))
    grad_weight = (grad * xhat).sum(dim=reduce_dims)
    scaled_grad = grad * w
    mean_scaled_grad_x = (scaled_grad * x).mean(dim=-1, keepdim=True)
    grad_input = scaled_grad * rstd - x * mean_scaled_grad_x * rstd.pow(3)
    return grad_input.to(dtype=input_.dtype), grad_weight.to(dtype=weight.dtype)


class _SGLangFusedQKNormRoPE(torch.autograd.Function):
    """SGLang fused QK-Norm+RoPE forward with RMSNorm+RoPE analytic backward."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        positions: torch.Tensor,
        eps: torch.Tensor,
        rotary_base: torch.Tensor,
    ):
        from sglang.jit_kernel.fused_qknorm_rope import fused_qk_norm_rope

        t = query.shape[0]
        nq, hd = query.shape[1], query.shape[2]
        nkv = key.shape[1]
        ctx.eps = float(eps.item())
        ctx.rotary_base = float(rotary_base.item())
        ctx.save_for_backward(query, key, q_weight, k_weight, positions)

        q_flat = query.reshape(t, nq * hd)
        k_flat = key.reshape(t, nkv * hd)
        v_flat = value.reshape(t, nkv * hd)
        qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1).contiguous()
        fused_qk_norm_rope(
            qkv,
            nq,
            nkv,
            nkv,
            hd,
            ctx.eps,
            q_weight.to(torch.bfloat16),
            k_weight.to(torch.bfloat16),
            ctx.rotary_base,
            True,
            positions,
            1.0,
            0.0,
            0.0,
            1.0,
        )
        q_size = nq * hd
        kv_size = nkv * hd
        q_out, k_out, v_out = qkv.split([q_size, kv_size, kv_size], dim=-1)
        return (
            q_out.reshape(t, nq, hd),
            k_out.reshape(t, nkv, hd),
            v_out.reshape(t, nkv, hd),
        )

    @staticmethod
    def backward(ctx, grad_query, grad_key, grad_value):
        query, key, q_weight, k_weight, positions = ctx.saved_tensors
        hd = query.shape[-1]
        cache = _get_rope_cache(
            query.device,
            hd,
            ctx.rotary_base,
            int(positions.max().item()) + 1,
        )
        gq_unrope = _neox_rope_backward(grad_query, cache, positions)
        gk_unrope = _neox_rope_backward(grad_key, cache, positions)
        dq, dq_w = _rmsnorm_backward(
            gq_unrope.reshape(-1, hd),
            query.reshape(-1, hd),
            q_weight,
            ctx.eps,
        )
        dk, dk_w = _rmsnorm_backward(
            gk_unrope.reshape(-1, hd),
            key.reshape(-1, hd),
            k_weight,
            ctx.eps,
        )
        return (
            dq.view_as(query),
            dk.view_as(key),
            grad_value,
            dq_w,
            dk_w,
            None,
            None,
            None,
        )


def fused_qk_norm_rope_with_grad(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    *,
    eps: float,
    rotary_base: float,
):
    eps_t = torch.tensor(float(eps), device=query.device)
    base_t = torch.tensor(float(rotary_base), device=query.device)
    return _SGLangFusedQKNormRoPE.apply(
        query, key, value, q_weight, k_weight, positions, eps_t, base_t
    )
# Sequences per backward chunk: bounds the live SDPA saved-tensor memory while
# amortizing autograd engine dispatch.  Chunk size does not affect gradient
# values: every sequence's backward is computed by the same per-seq SDPA
# kernels and accumulated into disjoint slices of the leaf gradients.
_FA_BWD_SEQ_CHUNK = 64


def _fa4_varlen_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    # One batched varlen call instead of the previous per-seq python loop.
    # The FA4 varlen kernel computes every sequence independently (per-seq
    # softmax, per-seq tiling), so batched output is bitwise identical to
    # per-seq calls; verified empirically on random ragged batches incl.
    # zero-length sequences (diff = 0 mismatched elements).
    from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if query.shape[0] == 0:
        return query.new_zeros(query.shape)

    # max_seqlen is a kernel-launch scalar, so the seq boundaries must be
    # read once.  One tolist() per tensor replaces the previous ~4n_seq
    # implicit device syncs inside the per-seq loop.
    cu_q_list = cu_q.tolist()
    cu_kv_list = cu_kv.tolist()
    max_seqlen_q = max(
        (end - start for start, end in zip(cu_q_list, cu_q_list[1:])),
        default=0,
    )
    max_seqlen_k = max(
        (end - start for start, end in zip(cu_kv_list, cu_kv_list[1:])),
        default=0,
    )
    out = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=float(softmax_scale),
        causal=True,
        softcap=0.0,
        # Pin the split-KV schedule: num_splits=1 keeps every sequence's
        # kernel schedule independent of batch composition (batch invariance).
        # This is currently the wrapper default; pass it explicitly so an
        # upstream default change cannot silently break the invariant.
        num_splits=1,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def _sdpa_one_seq(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    qh = query.transpose(0, 1).unsqueeze(0)
    kh = key.transpose(0, 1).unsqueeze(0)
    vh = value.transpose(0, 1).unsqueeze(0)
    n_q, n_kv = qh.shape[1], kh.shape[1]
    if n_q != n_kv:
        if n_q % n_kv != 0:
            raise RuntimeError(f"GQA head mismatch: q={n_q} kv={n_kv}")
        repeat = n_q // n_kv
        kh = kh.repeat_interleave(repeat, dim=1)
        vh = vh.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        dropout_p=0.0,
        is_causal=True,
        scale=float(softmax_scale),
    )
    return out.squeeze(0).transpose(0, 1)


def _fa_varlen_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_seq = cu_q.numel() - 1
    # Read boundaries once; per-seq int(cu_q[i]) calls each forced a device
    # sync (4 per sequence).
    cu_q_list = cu_q.tolist()
    cu_kv_list = cu_kv.tolist()

    # One set of graph leaves for the whole packed batch.  Slicing a leaf in
    # the per-seq loop lets autograd accumulate every chunk's gradient into
    # leaf.grad at exactly the positions the old per-seq copy-back wrote, so
    # the returned gradients stay bitwise identical while the autograd engine
    # is invoked once per chunk instead of once per sequence (~1900 -> ~30
    # times per layer).  Sequences are processed in bounded chunks to cap the
    # live SDPA saved-tensor memory.
    q_leaf = query.detach().requires_grad_(True)
    k_leaf = key.detach().requires_grad_(True)
    v_leaf = value.detach().requires_grad_(True)

    with torch.enable_grad():
        for chunk_start in range(0, n_seq, _FA_BWD_SEQ_CHUNK):
            chunk_end = min(chunk_start + _FA_BWD_SEQ_CHUNK, n_seq)
            outs = []
            grads = []
            for seq_idx in range(chunk_start, chunk_end):
                q0, q1 = cu_q_list[seq_idx], cu_q_list[seq_idx + 1]
                k0, k1 = cu_kv_list[seq_idx], cu_kv_list[seq_idx + 1]
                if q1 == q0:
                    continue
                outs.append(
                    _sdpa_one_seq(
                        q_leaf[q0:q1],
                        k_leaf[k0:k1],
                        v_leaf[k0:k1],
                        softmax_scale,
                    )
                )
                grads.append(grad_output[q0:q1].contiguous())
            if outs:
                torch.autograd.backward(outs, grads)

    dq = q_leaf.grad if q_leaf.grad is not None else torch.zeros_like(query)
    dk = k_leaf.grad if k_leaf.grad is not None else torch.zeros_like(key)
    dv = v_leaf.grad if v_leaf.grad is not None else torch.zeros_like(value)
    return dq, dk, dv


class _SGLangFA4Varlen(torch.autograd.Function):
    """FA4 varlen forward with SDPA backward (trainable GQA attention)."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
        softmax_scale: float,
    ):
        # Keep the scale as a host float: it is only a kernel scalar and the
        # previous CUDA-tensor round trip added one H2D copy plus one
        # blocking .item() sync per layer per direction.
        scale = float(softmax_scale)
        ctx.softmax_scale = scale
        ctx.save_for_backward(query, key, value, cu_q, cu_kv)
        return _fa4_varlen_forward(query, key, value, cu_q, cu_kv, scale)

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, cu_q, cu_kv = ctx.saved_tensors
        dq, dk, dv = _fa_varlen_backward(
            query,
            key,
            value,
            grad_output,
            cu_q,
            cu_kv,
            ctx.softmax_scale,
        )
        return dq, dk, dv, None, None, None


def fa4_varlen_with_grad(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    *,
    softmax_scale: float,
):
    scale = float(softmax_scale)
    return _SGLangFA4Varlen.apply(query, key, value, cu_q, cu_kv, scale)


class _IndexUnshuffle(torch.autograd.Function):
    """Scatter rank-concatenated THD tokens back to original packed order."""

    @staticmethod
    def forward(ctx, packed_by_rank: torch.Tensor, indices: torch.Tensor):
        ctx.save_for_backward(indices)
        out = packed_by_rank.new_empty(packed_by_rank.shape)
        out.index_copy_(0, indices, packed_by_rank)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indices,) = ctx.saved_tensors
        return grad_output.index_select(0, indices), None


def thd_cp_local_positions(
    cu_seqlens: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """RoPE positions for this CP rank's local THD tokens.

    ``cu_seqlens`` is the *global* packed layout (Megatron multiplies local
    lengths by ``cp_size``). Each local token gets its offset inside its
    original sequence, not ``arange(local_T)``.
    """
    from slime_plugins.models.qwen3_5_vl_utils import get_packed_cp_local_indices

    cu = cu_seqlens.to(device=device, dtype=torch.long)
    if cp_size <= 1:
        token_ids = torch.arange(int(cu[-1].item()), device=device, dtype=torch.long)
    else:
        token_ids = get_packed_cp_local_indices(cu, cp_size, cp_rank, device)
    seq_ids = torch.searchsorted(cu[1:], token_ids, right=True)
    return (token_ids - cu[seq_ids]).to(torch.int32)


def unshuffle_cp_rank_concat(
    packed_by_rank: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    cp_size: int,
) -> torch.Tensor:
    """Undo Megatron's two-chunk CP layout after an all-gather along the CP group."""
    from slime_plugins.models.qwen3_5_vl_utils import get_packed_cp_local_indices

    if cp_size <= 1:
        return packed_by_rank
    device = packed_by_rank.device
    indices = torch.cat(
        [get_packed_cp_local_indices(cu_seqlens, cp_size, rank, device) for rank in range(cp_size)]
    )
    full_len = int(cu_seqlens[-1].item() if torch.is_tensor(cu_seqlens) else cu_seqlens[-1])
    if indices.numel() != packed_by_rank.shape[0] or indices.numel() != full_len:
        raise RuntimeError(
            "THD CP unshuffle size mismatch: "
            f"packed={packed_by_rank.shape[0]} indices={indices.numel()} full={full_len}"
        )
    return _IndexUnshuffle.apply(packed_by_rank, indices)
