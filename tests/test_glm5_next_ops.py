import json

import pytest
import torch

from slime.utils.model_config import sglang_model_config_overrides
from slime_plugins.models.glm5_next.ops import mhc_post, mhc_pre, pooled_indexer_topk

NUM_GPUS = 0


@pytest.mark.parametrize("safe_gate,expected", [(True, -5.0), (False, None)])
def test_legacy_kda_gate_config(tmp_path, safe_gate, expected):
    config = {
        "model_type": "glm5_next",
        "text_config": {"linear_attn_config": {"safe_gate": safe_gate, "num_heads": 64}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    explicit = {"text_config": {"swiglu_limit": 10}, "max_position_embeddings": 2048}
    result = json.loads(sglang_model_config_overrides(tmp_path, json.dumps(explicit)))
    assert result["text_config"]["linear_attn_config"] == {
        "safe_gate": safe_gate,
        "num_heads": 64,
        "gate_lower_bound": expected,
    }
    assert result["text_config"]["swiglu_limit"] == 10
    assert result["max_position_embeddings"] == 2048


@pytest.mark.parametrize("lower_bound", [-3.0, None])
def test_explicit_kda_gate_config_takes_precedence(tmp_path, lower_bound):
    config = {"model_type": "glm5_next", "text_config": {"linear_attn_config": {"safe_gate": True}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    explicit = json.dumps({"text_config": {"linear_attn_config": {"gate_lower_bound": lower_bound}}})
    assert sglang_model_config_overrides(tmp_path, explicit) == explicit


def test_other_models_keep_server_overrides(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    explicit = '{"max_position_embeddings": 2048}'
    assert sglang_model_config_overrides(tmp_path, explicit) == explicit


@pytest.mark.parametrize("lengths,topk", [([1, 2, 3, 4, 5], 4), ([11, 7], 4), ([131, 19], 4), ([2053, 11], 2048)])
def test_pooling_preserves_causality_and_packed_boundaries(lengths, topk):
    # Increasing positive keys select the most recent full pools. Cover tails,
    # query chunk boundaries, and selection beyond the real 2048-token budget.
    n = sum(lengths)
    q = torch.ones(n, 1, 1)
    k = torch.arange(n, dtype=torch.float32)[:, None]
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    result = pooled_indexer_topk(q, k, torch.ones(n, 1), torch.zeros(n, 1), torch.zeros(4, 1), boundaries, topk, 4)
    start = 0
    for length in lengths:
        for position in range(length):
            complete_end = ((position + 1) // 4) * 4
            expected = set(range(start + max(0, complete_end - topk), start + position + 1))
            actual = result[start + position, 0]
            actual = actual[actual >= 0].tolist()
            assert len(actual) == len(set(actual))
            assert set(actual) == expected
        start += length


def test_learned_pool_gate_changes_selected_pool():
    q = torch.ones(8, 1, 1)
    k = torch.tensor([1, 1, 1, 20, 6, 6, 6, 6], dtype=torch.float32)[:, None]
    gates = torch.tensor([0, 0, 0, 30, 0, 0, 0, 0], dtype=torch.float32)[:, None]
    result = pooled_indexer_topk(q, k, torch.ones(8, 1), gates, torch.zeros(4, 1), torch.tensor([0, 8]), 4, 4)
    assert set(result[-1, 0, :4].tolist()) == {0, 1, 2, 3}


@pytest.mark.parametrize("cp_size", [2, 4])
def test_pooling_cp_selects_global_causal_keys(cp_size):
    # Each contiguous CP interval can cross a packed-sequence boundary or
    # begin/end inside a compression pool. It must retain global key indices.
    cu = torch.tensor([0, 131, 224])
    q = torch.ones(224, 1, 1)
    k = torch.arange(224, dtype=torch.float32)[:, None]
    weights, gates, ape = torch.ones(224, 1), torch.zeros(224, 1), torch.zeros(4, 1)
    full = pooled_indexer_topk(q, k, weights, gates, ape, cu, 16, 4)
    for rank in range(cp_size):
        first, last = rank * 224 // cp_size, (rank + 1) * 224 // cp_size
        actual = pooled_indexer_topk(q[first:last], k, weights[first:last], gates, ape, cu, 16, 4, query_start=first)
        torch.testing.assert_close(actual, full[first:last], rtol=0, atol=0)


def test_mhc_zero_mapping_and_backward():
    torch.manual_seed(42)
    x = torch.randn(3, 1, 16, requires_grad=True)
    weight = torch.zeros(24, 16, requires_grad=True)
    scale = torch.full((3,), 0.01, requires_grad=True)
    bias = torch.zeros(24, requires_grad=True)
    y, post, residual = mhc_pre(x, weight, scale, bias, 4, 1e-5, 1e-6, 20)
    expected = (x.reshape(3, 1, 4, 4) * (0.5 + 1e-6)).sum(-2)
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(post, torch.ones_like(post))
    torch.testing.assert_close(residual.sum(-1), torch.ones(3, 1, 4), atol=2e-6, rtol=0)
    output = mhc_post(y, x, post, residual)
    output.square().mean().backward()
    for parameter in (x, weight, scale, bias):
        assert parameter.grad is not None and parameter.grad.isfinite().all()
    assert weight.grad.abs().sum() > 0


def test_mhc_post_uses_source_to_destination_mixing():
    x = torch.tensor([[[2.0, 5.0, 7.0]]])
    mix = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]])
    result = mhc_post(torch.zeros(1, 1, 1), x, torch.zeros(1, 1, 3), mix)
    torch.testing.assert_close(result, torch.tensor([[[7.0, 2.0, 5.0]]]))


def test_mhc_backward_matches_numerical_derivatives():
    torch.manual_seed(81)
    tensors = (
        torch.randn(3, 1, 32, dtype=torch.float64),
        torch.randn(24, 32, dtype=torch.float64) * 0.2,
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        torch.randn(24, dtype=torch.float64) * 0.2,
    )

    def forward(x, weight, scale, bias):
        x, weight, scale, bias = (t.float() for t in (x, weight, scale, bias))
        y, post, residual = mhc_pre(x, weight, scale, bias, 4, 1e-5, 1e-6, 20)
        return mhc_post(y.sin(), x, post, residual)

    # The projection intentionally computes in FP32, so finite differences need
    # a larger perturbation than gradcheck's FP64 default to exceed roundoff.
    assert torch.autograd.gradcheck(
        forward, tuple(t.requires_grad_() for t in tensors), eps=1e-3, atol=5e-4, rtol=0.01, fast_mode=True
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
