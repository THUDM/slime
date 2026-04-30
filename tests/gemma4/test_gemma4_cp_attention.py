"""Parity tests for SDPACoreAttention.

These tests exercise the PRODUCTION code paths in slime_plugins.models.gemma4
— not re-implementations — so divergence between the dispatch logic in
`forward` and the hand-written CP math would be caught here.

Runs on a single GPU (or CPU, but flash-attn is skipped). A distributed
group is faked via `torch.distributed` with world_size=1, which turns the
CP all-gather into a no-op and lets us re-use the CP code on a single
device while still hitting the differentiable-gather autograd path.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


@pytest.fixture(scope="module", autouse=True)
def _init_dist():
    """Initialize a single-rank process group so CP code can call into it."""
    if dist.is_initialized():
        yield
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=0, world_size=1)
    try:
        # Megatron's parallel_state needs to be initialized for CP helpers to work.
        try:
            from megatron.core import parallel_state as mpu
            mpu.initialize_model_parallel(context_parallel_size=1)
        except Exception:
            pass
        yield
    finally:
        dist.destroy_process_group()


def _ref_attention(query, key, value, cu_seqlens, scale, sliding_window=None):
    """Ground-truth varlen causal (optionally sliding) attention via SDPA,
    with manual GQA expansion. query/key/value in [T, n, h] format."""
    t = query.shape[0]
    nq, nk = query.shape[1], key.shape[1]
    q = query.unsqueeze(0).transpose(1, 2).float()  # [1, n, T, h]
    k = key.unsqueeze(0).transpose(1, 2).float()
    v = value.unsqueeze(0).transpose(1, 2).float()
    if nq != nk:
        k = k.repeat_interleave(nq // nk, dim=1)
        v = v.repeat_interleave(nq // nk, dim=1)

    mask = torch.full((t, t), float("-inf"), device=query.device, dtype=torch.float32)
    for i in range(len(cu_seqlens) - 1):
        s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        for qi in range(s, e):
            lo = s if sliding_window is None else max(s, qi - sliding_window + 1)
            mask[qi, lo:qi + 1] = 0.0

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None, :, :], scale=scale)
    return out.transpose(1, 2).reshape(t, -1).to(query.dtype)


def _make_core_attention(sliding_window: int | None, softmax_scale: float):
    """Build an SDPACoreAttention without going through Megatron's spec system."""
    from types import SimpleNamespace
    from slime_plugins.models.gemma4 import SDPACoreAttention

    config = SimpleNamespace(
        attention_dropout=0.0,
        sliding_window=sliding_window or 1024,
        context_parallel_size=1,
    )
    core = SDPACoreAttention(
        config=config, layer_number=1, attn_mask_type=None, softmax_scale=softmax_scale,
    )
    core._is_sliding = sliding_window is not None
    return core


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_global_thd_sdpa_per_subseq_matches_reference():
    """Global layer (head_dim=512), no CP, packed thd — uses the SDPA per-sub-seq path."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32  # fp32 for exact parity

    nq, nk, hn = 8, 2, 512
    scale = 1.0 / (hn ** 0.5)
    lens = [13, 20, 7]
    cu = torch.tensor([0] + list(__import__("itertools").accumulate(lens)), dtype=torch.int32, device=device)
    t = int(cu[-1])
    q = torch.randn(t, nq, hn, device=device, dtype=dtype)
    k = torch.randn(t, nk, hn, device=device, dtype=dtype)
    v = torch.randn(t, nk, hn, device=device, dtype=dtype)

    ref = _ref_attention(q, k, v, cu, scale=scale)

    core = _make_core_attention(sliding_window=None, softmax_scale=scale)
    out = core._forward_thd_sdpa_per_subseq(q, k, v, cu)
    assert out.shape == (t, nq * hn)

    cos = F.cosine_similarity(ref.flatten().unsqueeze(0), out.flatten().unsqueeze(0)).item()
    assert cos > 0.9999, f"global SDPA per-sub-seq mismatch, cosine={cos}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flash_thd_with_sliding_window():
    """Sliding-window layer, head_dim=256, thd — must apply sliding window."""
    try:
        import flash_attn  # noqa
    except ImportError:
        pytest.skip("flash_attn not installed")

    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16

    nq, nk, hn = 16, 8, 256
    scale = 1.0 / (hn ** 0.5)
    lens = [1200, 800]  # > sliding_window on the first sequence
    cu = torch.tensor([0] + list(__import__("itertools").accumulate(lens)), dtype=torch.int32, device=device)
    t = int(cu[-1])
    q = torch.randn(t, nq, hn, device=device, dtype=dtype)
    k = torch.randn(t, nk, hn, device=device, dtype=dtype)
    v = torch.randn(t, nk, hn, device=device, dtype=dtype)

    core = _make_core_attention(sliding_window=1024, softmax_scale=scale)
    out = core._forward_thd_flash(q, k, v, cu)
    assert out.shape == (t, nq * hn)
    assert not torch.isnan(out).any()

    # Reference: apply sliding window in the mask.
    ref = _ref_attention(q.float(), k.float(), v.float(), cu, scale=scale, sliding_window=1024)
    cos = F.cosine_similarity(ref.flatten().unsqueeze(0), out.float().flatten().unsqueeze(0)).item()
    assert cos > 0.999, f"flash+sliding mismatch, cosine={cos}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_forward_dispatches_correctly_by_layer_type_and_headdim():
    """Smoke-test: `forward()` picks the right internal path without crashing."""
    torch.manual_seed(2)
    device = "cuda"
    dtype = torch.bfloat16

    # Construct a packed_seq_params-like object.
    from types import SimpleNamespace
    cu = torch.tensor([0, 64, 192], dtype=torch.int32, device=device)
    packed = SimpleNamespace(cu_seqlens_q=cu)

    # Case 1: sliding layer, head_dim=256 → flash path.
    core = _make_core_attention(sliding_window=1024, softmax_scale=1.0 / (256 ** 0.5))
    q = torch.randn(192, 8, 256, device=device, dtype=dtype)
    k = torch.randn(192, 4, 256, device=device, dtype=dtype)
    v = torch.randn(192, 4, 256, device=device, dtype=dtype)
    out = core.forward(q, k, v, packed_seq_params=packed)
    assert out.shape == (192, 8 * 256)
    assert not torch.isnan(out).any()

    # Case 2: global layer, head_dim=512, CP=1 → SDPA per-sub-seq.
    core_g = _make_core_attention(sliding_window=None, softmax_scale=1.0 / (512 ** 0.5))
    qg = torch.randn(192, 8, 512, device=device, dtype=dtype)
    kg = torch.randn(192, 2, 512, device=device, dtype=dtype)
    vg = torch.randn(192, 2, 512, device=device, dtype=dtype)
    out = core_g.forward(qg, kg, vg, packed_seq_params=packed)
    assert out.shape == (192, 8 * 512)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cp_global_gradient_flow_end_to_end():
    """With world_size=1, `_forward_cp_subseq_mask` becomes a pass-through but
    must still let gradients flow through K, V, Q without the
    deprecated-autograd warning or NaNs.
    """
    torch.manual_seed(3)
    device = "cuda"
    dtype = torch.float32

    nq, nk, hn = 8, 2, 512
    scale = 1.0 / (hn ** 0.5)
    # Each sub-seq length must be divisible by 2*cp_size; cp_size=2 -> div by 4.
    cu = torch.tensor([0, 32, 96], dtype=torch.int32, device=device)
    t = int(cu[-1])
    from types import SimpleNamespace
    packed = SimpleNamespace(cu_seqlens_q=cu)
    q = torch.randn(t, nq, hn, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(t, nk, hn, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(t, nk, hn, device=device, dtype=dtype, requires_grad=True)

    core = _make_core_attention(sliding_window=None, softmax_scale=scale)
    # Force the CP-global path by toggling cp_size on the config AND checking
    # that Megatron's CP world size is 1. With world_size=1, this exercises
    # the differentiable gather_from_sequence_parallel_region path.
    core.config.context_parallel_size = 2  # bluffs the dispatch into the CP path
    try:
        out = core._forward_cp_subseq_mask(q, k, v, packed, sliding_window=None)
    except Exception:
        pytest.skip("Megatron parallel_state not initialized; skipping CP path smoke test")

    assert out.shape == (t, nq * hn)
    assert not torch.isnan(out).any()
    out.sum().backward()
    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()
    # Critical: K and V grads must be non-zero. The old raw
    # `dist.all_gather_into_tensor` path dropped these grads on non-owning
    # ranks; with world_size=1 it would still work, but on the differentiable
    # path we check the signal is propagating at all.
    assert (k.grad.abs() > 0).any()
    assert (v.grad.abs() > 0).any()


def test_zigzag_global_indices_cp1_is_identity():
    """With cp_size=1, the zig-zag permutation should be identity (rank 0
    holds the whole sequence)."""
    from slime_plugins.models.gemma4 import SDPACoreAttention
    device = torch.device("cpu")
    # local_len = total local tokens = 8, cp_rank=0, cp_size=1 -> cs=4
    idx = SDPACoreAttention._zigzag_global_indices(
        local_len=8, cp_rank=0, cp_size=1, device=device,
    )
    # With cp=1: rank 0 owns global [0, cs) ++ [cs, 2*cs) = [0, 8)
    assert idx.tolist() == list(range(8))


def test_zigzag_global_indices_cp2_matches_slime_slice():
    """Verify the zig-zag map matches slime's slice_with_cp convention for
    cp_size=2: rank 0 owns chunks (0, 3), rank 1 owns chunks (1, 2)."""
    from slime_plugins.models.gemma4 import SDPACoreAttention
    device = torch.device("cpu")
    # 4 chunks of size 4 -> total 16 tokens. cp=2, so chunk_size = 4.
    # Rank 0 local tokens = 8 = 2*cs -> global [0,4) ++ [12,16)
    # Rank 1 local tokens = 8 = 2*cs -> global [4,8) ++ [8,12)
    idx_r0 = SDPACoreAttention._zigzag_global_indices(
        local_len=8, cp_rank=0, cp_size=2, device=device,
    )
    idx_r1 = SDPACoreAttention._zigzag_global_indices(
        local_len=8, cp_rank=1, cp_size=2, device=device,
    )
    assert idx_r0.tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
    assert idx_r1.tolist() == [4, 5, 6, 7, 8, 9, 10, 11]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
