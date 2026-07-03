import pytest
import torch

from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss

NUM_GPUS = 0


def test_policy_loss_keeps_finite_extreme_log_ratios_stable():
    ppo_kl = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    pg_losses, clipfrac = compute_policy_loss(
        ppo_kl,
        torch.ones_like(ppo_kl),
        eps_clip=0.2,
        eps_clip_high=0.2,
    )

    assert torch.isfinite(pg_losses).all()
    assert torch.isfinite(clipfrac).all()
    pg_losses.sum().backward()
    assert torch.isfinite(ppo_kl.grad).all()


def test_policy_loss_matches_unclamped_formula_for_healthy_ppo_kl():
    ppo_kl = torch.tensor([-0.1, 0.0, 0.1])
    advantages = torch.tensor([1.0, -2.0, 0.5])
    pg_losses, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip=0.2, eps_clip_high=0.2)

    ratio = (-ppo_kl).exp()
    expected1 = -ratio * advantages
    expected2 = -ratio.clamp(0.8, 1.2) * advantages
    torch.testing.assert_close(pg_losses, torch.maximum(expected1, expected2), rtol=0, atol=0)
    torch.testing.assert_close(clipfrac, torch.gt(expected2, expected1).float(), rtol=0, atol=0)


def test_low_var_kl_keeps_finite_extreme_log_ratios_stable():
    log_probs = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    kl = compute_approx_kl(log_probs, torch.zeros_like(log_probs), kl_loss_type="low_var_kl")

    assert torch.isfinite(kl).all()
    kl.sum().backward()
    assert torch.isfinite(log_probs.grad).all()


def test_nan_log_ratios_remain_visible():
    value = torch.tensor([float("nan")])
    pg_losses, _ = compute_policy_loss(value, torch.ones_like(value), eps_clip=0.2, eps_clip_high=0.2)
    kl = compute_approx_kl(value, torch.zeros_like(value), kl_loss_type="low_var_kl")

    assert torch.isnan(pg_losses).all()
    assert torch.isnan(kl).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
