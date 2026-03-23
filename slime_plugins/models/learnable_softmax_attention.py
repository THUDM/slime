"""
Flash attention with learnable softmax support for packed sequences (THD format).

Learnable softmax adds a per-head learnable offset to the attention logits before softmax,
acting as an attention "sink" that can absorb probability mass:
    P = softmax([QK^T/sqrt(d), offset])[:, :, :, :-1]

This is mathematically equivalent to:
    P = softmax(QK^T/sqrt(d)) * sigmoid(logsumexp(QK^T/sqrt(d)) - offset)

We use flash attention for the efficient QK^T computation and apply the sigmoid scaling
as a post-processing step, with a custom backward that correctly propagates gradients
through both the attention output and the logsumexp → sigmoid path.
"""

import math

import torch
from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_backward, flash_attn_varlen_func
from torch import Tensor


class _LearnableSoftmaxFlashAttnVarlen(torch.autograd.Function):
    """Custom autograd for flash attention with learnable softmax.

    Forward:
        out_vanilla, lse = flash_attn(q, k, v)
        scale = sigmoid(lse - offset)        # per head per token
        out = scale * out_vanilla

    Backward:
        Pass d_out' = scale * d_out and out' = scale * out_vanilla to flash_attn backward.
        This correctly accounts for the gradient through lse because:
            di' = d_out' . out' = scale^2 * (d_out . out_vanilla)
        which gives the correct total derivative d_S = P * scale * (D - scale * di).
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_offset,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        dropout_p,
        deterministic,
    ):
        out_vanilla, softmax_lse, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(window_size_left, window_size_right),
            deterministic=deterministic,
            return_attn_probs=True,
        )
        # softmax_lse: (nheads, total_q) in fp32
        # softmax_offset: (nheads,)
        scale = torch.sigmoid(softmax_lse - softmax_offset.unsqueeze(1))  # (nheads, total_q)
        scale_t = scale.t().contiguous()  # (total_q, nheads)
        # Keep output in input dtype for flash_attn backward compatibility
        out_learn = (out_vanilla.float() * scale_t.unsqueeze(-1)).to(q.dtype)

        ctx.save_for_backward(
            q,
            k,
            v,
            out_vanilla,
            out_learn,
            softmax_lse,
            softmax_offset,
            scale_t,
            cu_seqlens_q,
            cu_seqlens_k,
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.dropout_p = dropout_p
        ctx.deterministic = deterministic
        return out_learn

    @staticmethod
    def backward(ctx, d_out):
        (
            q,
            k,
            v,
            out_vanilla,
            out_learn,
            softmax_lse,
            softmax_offset,
            scale_t,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = ctx.saved_tensors

        # Scale d_out for flash_attn backward: d_out' = scale * d_out
        d_out_modified = (d_out.float() * scale_t.unsqueeze(-1)).to(q.dtype)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        head_size_og = d_out_modified.size(2)
        d_out_padded = d_out_modified
        if head_size_og % 8 != 0:
            d_out_padded = torch.nn.functional.pad(d_out_modified, [0, 8 - head_size_og % 8])

        _wrapped_flash_attn_varlen_backward(
            d_out_padded.contiguous(),
            q,
            k,
            v,
            out_learn.contiguous(),
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size_left,
            ctx.window_size_right,
            0.0,  # softcap
            None,  # alibi_slopes
            ctx.deterministic,
            rng_state=None,
        )

        dq = dq[..., :head_size_og]
        dk = dk[..., :head_size_og]
        dv = dv[..., :head_size_og]

        # Gradient for softmax_offset
        scale = scale_t.t()  # (nheads, total_q)
        dot_product = (d_out.float() * out_vanilla.float()).sum(-1).t()  # (nheads, total_q)
        sigmoid_grad = scale * (1 - scale)
        d_offset = -(dot_product * sigmoid_grad).sum(1)  # (nheads,)

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            d_offset,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def learnable_softmax_flash_attn_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_offset: Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    dropout_p: float = 0.0,
    deterministic: bool = False,
) -> Tensor:
    """Flash attention with learnable softmax for variable-length packed sequences.

    Args:
        q: (total_q, nheads, headdim) query tensor
        k: (total_k, nheads_k, headdim) key tensor
        v: (total_k, nheads_k, headdim) value tensor
        cu_seqlens_q: (batch_size + 1,) cumulative sequence lengths for queries
        cu_seqlens_k: (batch_size + 1,) cumulative sequence lengths for keys
        max_seqlen_q: max sequence length in queries
        max_seqlen_k: max sequence length in keys
        softmax_offset: (nheads,) learnable per-head offset
        softmax_scale: scaling factor for QK^T (default: 1/sqrt(headdim))
        causal: whether to use causal attention mask
        window_size: (left, right) sliding window sizes, -1 for no window
        dropout_p: dropout probability
        deterministic: use deterministic backward

    Returns:
        out: (total_q, nheads, headdim) attention output
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))
    return _LearnableSoftmaxFlashAttnVarlen.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_offset,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        dropout_p,
        deterministic,
    )
