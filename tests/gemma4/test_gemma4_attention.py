from types import SimpleNamespace

import pytest
import torch

try:
    from slime_plugins.models.gemma4 import Gemma4SelfAttention, VNorm
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if not (missing == "megatron" or missing.startswith("megatron.") or missing == "mbridge"):
        raise
    from tests.gemma4._standalone_imports import load_gemma4_model_module

    _gemma4 = load_gemma4_model_module()
    Gemma4SelfAttention = _gemma4.Gemma4SelfAttention
    VNorm = _gemma4.VNorm


def _stub_attention(num_attention_heads, num_kv_heads, head_dim, hidden_size):
    attn = object.__new__(Gemma4SelfAttention)
    torch.nn.Module.__init__(attn)

    q_per_kv = num_attention_heads // num_kv_heads
    out_width = num_kv_heads * (q_per_kv + 2) * head_dim
    linear_qkv = torch.nn.Linear(hidden_size, out_width, bias=False)
    torch.nn.init.normal_(linear_qkv.weight, std=0.02)

    def _linear_qkv(h):
        return linear_qkv(h), None

    attn.linear_qkv = _linear_qkv
    attn.num_attention_heads_per_partition = num_attention_heads
    attn.num_query_groups_per_partition = num_kv_heads
    attn.hidden_size_per_attention_head = head_dim
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

    mixed, _ = attn.linear_qkv(hidden)
    q_per_kv = num_attention_heads // num_kv_heads
    mixed = mixed.view(seq_len, batch, num_kv_heads, (q_per_kv + 2) * head_dim)
    q_width = q_per_kv * head_dim
    raw_q, raw_k, _raw_v = torch.split(mixed, [q_width, head_dim, head_dim], dim=3)
    raw_q = raw_q.reshape(seq_len, batch, -1, head_dim)

    expected_query = attn.q_layernorm(raw_q)
    expected_key = attn.k_layernorm(raw_k)
    expected_value = attn.v_norm(raw_k)

    assert torch.allclose(query, expected_query), "query mismatch"
    assert torch.allclose(key, expected_key), "key must be k_norm(raw_k)"
    assert torch.allclose(value, expected_value), (
        "value must be v_norm(raw_k); if this fails, v is being derived from " "k_norm(raw_k) instead of raw_k"
    )


def test_global_k_eq_v_does_not_mutate_k_layernorm():
    torch.manual_seed(1)
    attn, _ = _stub_attention(8, 2, 512, 256)
    attn._is_global = True

    k_layernorm_before = attn.k_layernorm
    hidden = torch.randn(3, 1, 256)
    _ = attn.get_query_key_value_tensors(hidden)
    assert attn.k_layernorm is k_layernorm_before


def test_global_k_eq_v_rejects_output_gate():
    attn, _ = _stub_attention(8, 2, 512, 256)
    attn._is_global = True
    with pytest.raises(NotImplementedError):
        attn.get_query_key_value_tensors(torch.randn(3, 1, 256), output_gate=True)


def test_sliding_layer_applies_v_norm_to_value():
    torch.manual_seed(2)
    num_attention_heads, num_kv_heads, head_dim, hidden_size = 8, 2, 256, 256
    attn, linear_qkv = _stub_attention(num_attention_heads, num_kv_heads, head_dim, hidden_size)
    attn._is_global = False

    seq_len, batch = 3, 1
    raw_q = torch.randn(seq_len, batch, num_attention_heads, head_dim)
    raw_k = torch.randn(seq_len, batch, num_kv_heads, head_dim)
    raw_v = torch.randn(seq_len, batch, num_kv_heads, head_dim)

    def _fake_parent(*_a, **_k):
        return raw_q, raw_k, raw_v

    import unittest.mock as mock

    _Base = Gemma4SelfAttention.__mro__[1]
    with mock.patch.object(_Base, "get_query_key_value_tensors", _fake_parent):
        query, key, value = attn.get_query_key_value_tensors(torch.randn(seq_len, batch, hidden_size))

    assert torch.equal(query, raw_q)
    assert torch.equal(key, raw_k)
    assert torch.allclose(value, attn.v_norm(raw_v))
