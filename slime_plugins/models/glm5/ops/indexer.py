import os

import torch

# Bind flashinfer.comm's module-level `cudart = CudaRTLibrary()` to the *real*
# libcudart before tilelang loads its `libcudart_stub.so`.  On a cu12 box whose
# flashinfer pulled in cu13 deps, importing flashinfer.comm *after* the stub is
# resident makes `find_loaded_library("libcudart")` match the stub (which lacks
# `cudaDeviceReset`), crashing the lazy `dsa_indexer` import inside the forward.
# Importing it here — while only the real libcudart is loaded — caches the
# module with a valid binding; later imports are no-ops.  Best-effort: other
# environments without flashinfer must still import this module.
try:  # noqa: SIM105
    import flashinfer.comm  # noqa: F401
except Exception:
    pass

from .tilelang_indexer_bwd import indexer_bwd_interface
from .tilelang_indexer_fwd import indexer_fwd_interface


def _sglang_fp8_indexer_logits(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Evaluate GLM5 indexer logits with SGLang's FP8 DSA kernels."""

    if index_q.shape[-1] != 128:
        raise ValueError("SGLang FP8 indexer alignment requires head_dim=128, " f"got {index_q.shape[-1]}")
    import deep_gemm
    from sglang.jit_kernel.fused_store_index_cache import fused_store_index_k_cache
    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
    from sglang.srt.layers.attention.dsa.triton_kernel import act_quant

    q_rotated = rotate_activation(index_q.contiguous())
    if index_k.ndim == 3:
        if index_k.shape[1] != 1:
            raise ValueError(f"Expected one indexer KV head, got {index_k.shape}")
        index_k = index_k.squeeze(1)
    k_rotated = rotate_activation(index_k.contiguous())

    q_fp8, q_scale = act_quant(q_rotated, 128, "ue8m0")
    page_size = 64
    num_k = k_rotated.shape[0]
    num_pages = (num_k + page_size - 1) // page_size
    packed_k = torch.empty(
        (num_pages, page_size * (128 + 4)),
        dtype=torch.uint8,
        device=k_rotated.device,
    )
    fused_store_index_k_cache(
        k_rotated,
        packed_k,
        torch.arange(num_k, dtype=torch.int64, device=k_rotated.device),
        page_size=page_size,
    )
    k_fp8 = packed_k[:, : page_size * 128].contiguous().view(torch.float8_e4m3fn).reshape(-1, 128)[:num_k]
    k_scale = packed_k[:, page_size * 128 :].contiguous().view(torch.float32).reshape(-1)[:num_k]
    scaled_weights = weights.float() * q_scale.squeeze(-1).float()
    scaled_weights = (scaled_weights * (index_q.shape[-1] ** -0.5)).contiguous()
    logits = deep_gemm.fp8_mqa_logits(
        q_fp8,
        (k_fp8, k_scale),
        scaled_weights,
        starts.to(torch.int32).contiguous(),
        ends.to(torch.int32).contiguous(),
        clean_logits=False,
    )
    key_positions = torch.arange(num_k, dtype=torch.int32, device=index_q.device).unsqueeze(0)
    valid = (key_positions >= starts.to(torch.int32).unsqueeze(1)) & (
        key_positions < ends.to(torch.int32).unsqueeze(1)
    )
    return logits.masked_fill(~valid, float("-inf"))


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


def pytorch_topk_with_invalid_padding(logits: torch.Tensor, topk: int):
    """Select up to ``topk`` keys and retain the fixed-width DSA layout.

    Short packed microbatches can contain fewer total keys than the model's
    configured DSA top-k.  SGLang represents the missing selections with -1;
    mirror that representation instead of passing an out-of-range ``k`` to
    ``torch.topk``.
    """
    selected = min(topk, logits.shape[-1])
    scores, indices = torch.topk(logits, selected, dim=-1)
    indices = indices.to(torch.int32)
    indices = indices.masked_fill(scores == -torch.inf, -1)
    if selected == topk:
        return scores, indices

    pad_shape = (*logits.shape[:-1], topk - selected)
    scores = torch.cat(
        (scores, logits.new_full(pad_shape, float("-inf"))),
        dim=-1,
    )
    indices = torch.cat(
        (
            indices,
            torch.full(
                pad_shape,
                -1,
                dtype=torch.int32,
                device=logits.device,
            ),
        ),
        dim=-1,
    )
    return scores, indices


class IndexerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
        topk_indices: torch.Tensor | None = None,
    ):
        _, head_num, _ = index_q.shape
        if os.getenv("MEGATRON_USE_SGLANG_FP8_INDEXER", "0") == "1":
            logits = _sglang_fp8_indexer_logits(
                index_q,
                index_k,
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
            )
        else:
            logits = indexer_fwd_interface(
                index_q,
                index_k,
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
                clean_logits=True,
            )
        if topk_indices is None:
            index_score, topk_indices = pytorch_topk_with_invalid_padding(logits, topk)
            if os.getenv("MEGATRON_USE_SGLANG_FP8_INDEXER", "0") == "1":
                invalid_sort_key = torch.iinfo(torch.int32).max
                topk_indices = torch.sort(
                    topk_indices.masked_fill(topk_indices < 0, invalid_sort_key),
                    dim=-1,
                ).values
                topk_indices = topk_indices.masked_fill(topk_indices == invalid_sort_key, -1)

        index_score = pytorch_extract_topk_scores(logits, topk_indices)

        ctx.save_for_backward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices)
        ctx.topk = topk
        ctx.head_num = head_num
        return index_score, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, grad_indices):
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices = ctx.saved_tensors
        grad_q, grad_w, grad_k = indexer_bwd_interface(index_q, weights, index_k, topk_indices, grad_scores)
        return grad_q, grad_k, grad_w, None, None, None, None, None, None, None


def lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    topk_indices: torch.Tensor | None = None,
):
    return IndexerFunction.apply(index_q, index_k, weights.squeeze(-1), cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices)


def generate_varlen_mask_params(cu_seqlens):
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends
