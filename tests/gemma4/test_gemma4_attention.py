"""Unit tests for ``Gemma4SelfAttention.get_query_key_value_tensors``.

These tests target the two code paths independently:

- **Global K=V**: ``_split_qkv_global_k_eq_v`` must produce
  ``key = k_norm(raw_k)`` and ``value = v_norm(raw_k)`` without mutating
  ``self.k_layernorm`` and without going through the parent class. We
  construct a minimal stub of the parent attribs that the method reads.

- **Sliding**: delegates to the parent, and then applies ``v_norm`` to V.
  Covered indirectly in ``test_gemma4_cp_attention.py``; here we only verify that
  the non-global path produces a V tensor that equals ``v_norm(raw_v)``.
"""

from types import SimpleNamespace

import pytest
import torch

from slime_plugins.models.gemma4 import Gemma4SelfAttention, VNorm


def _stub_attention(num_attention_heads, num_kv_heads, head_dim, hidden_size):
    """Build a Gemma4SelfAttention-compatible stub without invoking
    ``SelfAttention.__init__`` (which needs TE / process group / etc.).

    We populate only the attributes that ``_split_qkv_global_k_eq_v`` and
    ``get_query_key_value_tensors`` read.
    """
    attn = object.__new__(Gemma4SelfAttention)
    # Gemma4SelfAttention is a nn.Module (through SelfAttention); we must
    # initialize the Module machinery before assigning submodules.
    torch.nn.Module.__init__(attn)

    # Deterministic linear_qkv: output width = num_kv_heads * (q_per_kv + 2) * head_dim
    q_per_kv = num_attention_heads // num_kv_heads
    out_width = num_kv_heads * (q_per_kv + 2) * head_dim
    linear_qkv = torch.nn.Linear(hidden_size, out_width, bias=False)
    torch.nn.init.normal_(linear_qkv.weight, std=0.02)

    def _linear_qkv(h):
        # Megatron's ColumnParallelLinear returns (out, bias); mimic that.
        return linear_qkv(h), None

    attn.linear_qkv = _linear_qkv
    attn.num_attention_heads_per_partition = num_attention_heads
    attn.num_query_groups_per_partition = num_kv_heads
    attn.hidden_size_per_attention_head = head_dim
    # Learnable k_norm / q_norm so their effect is visible.
    attn.q_layernorm = torch.nn.LayerNorm(head_dim)
    attn.k_layernorm = torch.nn.LayerNorm(head_dim)
    attn.v_norm = VNorm(head_dim, eps=1e-6)
    attn.config = SimpleNamespace(
        layernorm_epsilon=1e-6,
        attention_k_eq_v=True,
    )
    attn._is_global = False  # flipped per-test
    return attn, linear_qkv


def test_global_k_eq_v_produces_k_norm_and_v_norm_of_raw_k():
    """With _is_global=True, value must equal v_norm(raw_k_proj) — NOT
    v_norm(k_norm(raw_k_proj)). This guards the bug where the parent-class
    path would pre-normalize K before we extract V from it."""
    torch.manual_seed(0)
    num_attention_heads, num_kv_heads, head_dim, hidden_size = 8, 2, 512, 256
    attn, linear_qkv = _stub_attention(num_attention_heads, num_kv_heads, head_dim, hidden_size)
    attn._is_global = True

    seq_len, batch = 4, 1
    hidden = torch.randn(seq_len, batch, hidden_size)

    query, key, value = attn.get_query_key_value_tensors(hidden)

    assert query.shape == (seq_len, batch, num_attention_heads, head_dim)
    assert key.shape == (seq_len, batch, num_kv_heads, head_dim)
    assert value.shape == (seq_len, batch, num_kv_heads, head_dim)

    # Recompute expected tensors independently. We use the same linear_qkv
    # weights + norms to derive ground truth.
    mixed, _ = attn.linear_qkv(hidden)
    q_per_kv = num_attention_heads // num_kv_heads
    mixed = mixed.view(seq_len, batch, num_kv_heads, (q_per_kv + 2) * head_dim)
    q_width = q_per_kv * head_dim
    raw_q, raw_k, _raw_v = torch.split(mixed, [q_width, head_dim, head_dim], dim=3)
    raw_q = raw_q.reshape(seq_len, batch, -1, head_dim)

    expected_query = attn.q_layernorm(raw_q)
    expected_key = attn.k_layernorm(raw_k)
    expected_value = attn.v_norm(raw_k)  # v_norm applied to RAW k, not k_norm(raw_k)

    assert torch.allclose(query, expected_query), "query mismatch"
    assert torch.allclose(key, expected_key), "key must be k_norm(raw_k)"
    assert torch.allclose(value, expected_value), (
        "value must be v_norm(raw_k) — if this fails, v is being derived from "
        "k_norm(raw_k) instead of raw_k"
    )


def test_global_k_eq_v_does_not_mutate_k_layernorm():
    """Before the refactor, the implementation set ``self.k_layernorm = None``
    around the parent call. Any exception in the parent would leak that
    mutation permanently, and concurrent construction of sibling layers
    would race. Confirm the attribute is untouched across a forward."""
    torch.manual_seed(1)
    attn, _ = _stub_attention(8, 2, 512, 256)
    attn._is_global = True

    k_layernorm_before = attn.k_layernorm
    hidden = torch.randn(3, 1, 256)
    _ = attn.get_query_key_value_tensors(hidden)
    assert attn.k_layernorm is k_layernorm_before


def test_global_k_eq_v_rejects_output_gate():
    """output_gate is incompatible with the K=V split (the parent class uses
    a different tensor shape for gated attention). Ensure we fail loudly."""
    attn, _ = _stub_attention(8, 2, 512, 256)
    attn._is_global = True
    with pytest.raises(NotImplementedError):
        attn.get_query_key_value_tensors(torch.randn(3, 1, 256), output_gate=True)


def test_sliding_layer_applies_v_norm_to_value():
    """For non-global (sliding) layers, value comes from the standard QKV
    split and then gets v_norm applied. Verify the non-K=V path."""
    torch.manual_seed(2)
    num_attention_heads, num_kv_heads, head_dim, hidden_size = 8, 2, 256, 256
    attn, linear_qkv = _stub_attention(num_attention_heads, num_kv_heads, head_dim, hidden_size)
    attn._is_global = False  # sliding — parent path + v_norm

    # Patch the parent's call to return a known (q, k, v) triple. We avoid
    # calling the real SelfAttention.get_query_key_value_tensors (needs full
    # Megatron init) by monkey-patching just the bound method lookup.
    seq_len, batch = 3, 1
    raw_q = torch.randn(seq_len, batch, num_attention_heads, head_dim)
    raw_k = torch.randn(seq_len, batch, num_kv_heads, head_dim)
    raw_v = torch.randn(seq_len, batch, num_kv_heads, head_dim)

    def _fake_parent(*_a, **_k):
        return raw_q, raw_k, raw_v

    # Call the method directly via the base-class path we want to test. We
    # cannot use super() without a real SelfAttention, so re-execute the
    # split logic as written in get_query_key_value_tensors by calling with
    # monkey-patched super().
    import unittest.mock as mock
    from megatron.core.transformer.attention import SelfAttention as _Base
    with mock.patch.object(_Base, "get_query_key_value_tensors", _fake_parent):
        query, key, value = attn.get_query_key_value_tensors(
            torch.randn(seq_len, batch, hidden_size)
        )

    assert torch.equal(query, raw_q)
    assert torch.equal(key, raw_k)
    assert torch.allclose(value, attn.v_norm(raw_v))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
