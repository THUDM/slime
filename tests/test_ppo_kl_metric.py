import sys
import types
from argparse import Namespace
from decimal import Decimal, localcontext

import pytest
import torch

from slime.utils.ppo_utils import compute_approx_kl

NUM_GPUS = 0


@pytest.mark.parametrize("kl_loss_type", ["k3", "low_var_kl"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("base_log_probability", [-3.0, -0.1])
def test_near_policy_kl_preserves_small_nonnegative_values(kl_loss_type, weighted, base_log_probability):
    base = torch.full((7,), base_log_probability, dtype=torch.float32)
    log_probs = (
        base + torch.tensor([-1e-3, -1e-4, -1e-5, 0.0, 1e-5, 1e-4, 1e-3], dtype=torch.float32)
    ).requires_grad_()
    weights = (
        torch.tensor([1.0, 1.25, 2.0, 1.0, 1.25, 2.0, 1.0], dtype=torch.float32) if weighted else torch.ones_like(base)
    )
    if weighted:
        old_log_probs = log_probs.detach() - weights.log()
        weights = (log_probs - old_log_probs).exp()
    kl = compute_approx_kl(log_probs, base, kl_loss_type, weights if weighted else None)
    log_ratio = -(log_probs.detach() - base)

    with localcontext() as context:
        context.prec = 80
        expected = []
        for value, weight in zip(log_ratio.tolist(), weights.tolist(), strict=True):
            x = Decimal.from_float(value)
            expected.append(float((x.exp() - 1 - x) * Decimal.from_float(weight)))
    expected = torch.tensor(expected, dtype=torch.float64)
    # Near zero, the rounding budget should scale with x, not with exp(x)'s
    # O(1) term. Compare at the represented inputs, without quantizing the oracle.
    rounding_budget = (2 * torch.finfo(torch.float32).eps * log_ratio.abs() * weights).double()
    assert torch.all(kl >= 0)
    assert torch.all((kl.double() - expected).abs() <= rounding_budget)

    gradient = torch.autograd.grad(kl.sum(), log_probs)[0]
    expected_gradient = (
        -log_ratio.double() if weighted else -torch.expm1(log_ratio.double())
    ) * weights.detach().double()
    assert torch.isfinite(gradient).all()
    assert torch.all((gradient.double() - expected_gradient).abs() <= 2 * torch.finfo(torch.float32).eps * weights)


@pytest.mark.parametrize("kl_loss_type", ["k3", "low_var_kl"])
def test_kl_importance_weighting_precedes_low_variance_clamp(kl_loss_type):
    weights = torch.tensor([0.01, 1.0, 2.0], dtype=torch.float32)
    actual = compute_approx_kl(
        torch.full((3,), -6.0, dtype=torch.float32), torch.full((3,), -1.0, dtype=torch.float32), kl_loss_type, weights
    )
    expected = (torch.expm1(torch.tensor(5.0, dtype=torch.float64)) - 5.0) * weights.double()
    if kl_loss_type == "low_var_kl":
        expected = expected.clamp(min=-10, max=10)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=0.0)


def test_ppo_estimator_does_not_corrupt_logged_kl(monkeypatch):
    previous_loss = sys.modules.pop("slime.backends.megatron_utils.loss", None)
    previous_cp_utils = sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        is_pipeline_last_stage=lambda: True,
    )
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = mpu_stub
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)

    try:
        from slime.backends.megatron_utils.loss import compute_advantages_and_returns

        log_probs = [torch.tensor([0.5, 0.7, 0.9])]
        ref_log_probs = [torch.tensor([0.4, 0.5, 0.6])]
        expected_kl = compute_approx_kl(log_probs[0], ref_log_probs[0], kl_loss_type="k1")
        rollout_data = {
            "log_probs": log_probs,
            "ref_log_probs": ref_log_probs,
            "rewards": [1.0],
            "values": [torch.zeros(3)],
            "response_lengths": [3],
            "total_lengths": [5],
            "loss_masks": [torch.ones(3)],
        }
        args = Namespace(
            advantage_estimator="ppo",
            kl_coef=0.05,
            kl_loss_type="k1",
            use_rollout_logprobs=False,
            custom_advantage_function_path=None,
            normalize_advantages=False,
            use_opd=False,
            gamma=1.0,
            lambd=1.0,
        )
        compute_advantages_and_returns(args, rollout_data)
        torch.testing.assert_close(rollout_data["kl"][0], expected_kl)
    finally:
        if previous_loss is None:
            sys.modules.pop("slime.backends.megatron_utils.loss", None)
        else:
            sys.modules["slime.backends.megatron_utils.loss"] = previous_loss
        if previous_cp_utils is None:
            sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
        else:
            sys.modules["slime.backends.megatron_utils.cp_utils"] = previous_cp_utils


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
