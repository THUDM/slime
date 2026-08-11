from types import SimpleNamespace

import pytest
import torch

from slime_plugins.models.glm5.glm5 import DSAMLASelfAttention, IdentityOp

NUM_GPUS = 0


class _FusedQUp(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.layer_norm_weight = torch.nn.Parameter(weight.clone())


def _make_attention(weight: torch.Tensor, *, zero_centered_gamma: bool = False) -> DSAMLASelfAttention:
    attention = DSAMLASelfAttention.__new__(DSAMLASelfAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-5,
        layernorm_zero_centered_gamma=zero_centered_gamma,
    )
    attention.q_layernorm = IdentityOp()
    attention.linear_q_up_proj = _FusedQUp(weight)
    return attention


def test_glm5_indexer_uses_fused_q_rmsnorm_without_upstream_gradients():
    raw_q = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [0.25, 0.5, -0.75, 1.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16)
    attention = _make_attention(weight)

    actual = attention._get_indexer_q_input(raw_q)
    expected = torch.nn.functional.rms_norm(
        raw_q.detach().float(),
        normalized_shape=(raw_q.shape[-1],),
        weight=weight.float(),
        eps=attention.config.layernorm_epsilon,
    ).to(raw_q.dtype)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(actual, raw_q.detach())
    assert not actual.requires_grad

    wq_b = torch.nn.Linear(raw_q.shape[-1], 3, bias=False, dtype=torch.bfloat16)
    wq_b(actual).float().sum().backward()
    assert wq_b.weight.grad is not None
    assert raw_q.grad is None
    assert attention.linear_q_up_proj.layer_norm_weight.grad is None


def test_glm5_indexer_q_rmsnorm_supports_zero_centered_gamma_and_unfused_norm():
    raw_q = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.bfloat16)
    stored_weight = torch.tensor([-0.5, 0.0, 0.5, 1.0], dtype=torch.bfloat16)
    attention = _make_attention(stored_weight, zero_centered_gamma=True)

    actual = attention._get_indexer_q_input(raw_q)
    expected = torch.nn.functional.rms_norm(
        raw_q.float(),
        normalized_shape=(raw_q.shape[-1],),
        weight=stored_weight.float() + 1.0,
        eps=attention.config.layernorm_epsilon,
    ).to(raw_q.dtype)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    del attention.linear_q_up_proj.layer_norm_weight
    attention.q_layernorm = torch.nn.RMSNorm(raw_q.shape[-1], eps=attention.config.layernorm_epsilon).to(
        dtype=torch.bfloat16
    )
    unfused = attention._get_indexer_q_input(raw_q)
    torch.testing.assert_close(unfused, attention.q_layernorm(raw_q), rtol=0.0, atol=0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
