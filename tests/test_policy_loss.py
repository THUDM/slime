import pytest
import torch

from slime.utils.ppo_utils import compute_policy_loss


def _ppo_kl_from_ratio(ratio: torch.Tensor) -> torch.Tensor:
    return -ratio.log()


@pytest.mark.unit
def test_clipped_policy_loss_matches_existing_formula():
    ratio = torch.tensor([1.5, 0.7, 1.1, 0.9], dtype=torch.float32)
    advantages = torch.tensor([2.0, -3.0, -1.0, 0.5], dtype=torch.float32)
    eps_clip = 0.2
    eps_clip_high = 0.3

    pg_loss, clipfrac, aux = compute_policy_loss(
        _ppo_kl_from_ratio(ratio),
        advantages,
        eps_clip,
        eps_clip_high,
    )

    expected_loss = torch.maximum(
        -ratio * advantages,
        -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages,
    )
    expected_clipfrac = torch.gt(
        -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages,
        -ratio * advantages,
    ).float()

    torch.testing.assert_close(pg_loss, expected_loss)
    torch.testing.assert_close(clipfrac, expected_clipfrac)
    assert aux == {}


@pytest.mark.unit
def test_sapo_policy_loss_matches_soft_gate_formula():
    ratio = torch.tensor([1.5, 0.7, 1.0, 1.4], dtype=torch.float32)
    advantages = torch.tensor([2.0, -3.0, 0.0, -1.0], dtype=torch.float32)
    eps_clip = 0.2
    eps_clip_high = 0.3
    tau_pos = 1.0
    tau_neg = 1.05

    pg_loss, clipfrac, aux = compute_policy_loss(
        _ppo_kl_from_ratio(ratio),
        advantages,
        eps_clip,
        eps_clip_high,
        policy_loss_type="sapo",
        sapo_tau_pos=tau_pos,
        sapo_tau_neg=tau_neg,
    )

    tau = torch.where(
        advantages > 0,
        torch.full_like(advantages, tau_pos),
        torch.full_like(advantages, tau_neg),
    )
    expected_soft_ratio = torch.sigmoid(tau * (ratio - 1.0)) * (4.0 / tau)
    expected_loss = -expected_soft_ratio * advantages
    expected_clipfrac = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    torch.testing.assert_close(pg_loss, expected_loss)
    torch.testing.assert_close(clipfrac, expected_clipfrac)
    torch.testing.assert_close(aux["sapo_soft_ratio"], expected_soft_ratio)
    assert set(aux) == {"sapo_soft_ratio"}


@pytest.mark.unit
def test_sapo_gradient_matches_unclipped_policy_gradient_at_unit_ratio():
    ppo_kl = torch.zeros(4, dtype=torch.float32, requires_grad=True)
    advantages = torch.tensor([2.0, -3.0, 0.0, 0.5], dtype=torch.float32)

    pg_loss, _, _ = compute_policy_loss(
        ppo_kl,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.2,
        policy_loss_type="sapo",
        sapo_tau_pos=1.0,
        sapo_tau_neg=1.05,
    )
    pg_loss.sum().backward()

    torch.testing.assert_close(ppo_kl.grad, advantages)


@pytest.mark.unit
def test_sapo_policy_loss_is_finite_for_large_log_ratios():
    ppo_kl = torch.tensor([-100.0, 100.0], dtype=torch.float32, requires_grad=True)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)

    pg_loss, clipfrac, aux = compute_policy_loss(
        ppo_kl,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.2,
        policy_loss_type="sapo",
        sapo_tau_pos=1.0,
        sapo_tau_neg=1.05,
    )
    pg_loss.sum().backward()

    assert torch.isfinite(pg_loss).all()
    assert torch.isfinite(clipfrac).all()
    assert torch.isfinite(aux["sapo_soft_ratio"]).all()
    assert torch.isfinite(ppo_kl.grad).all()


@pytest.mark.unit
def test_unknown_policy_loss_type_raises():
    with pytest.raises(ValueError, match="Unknown policy_loss_type"):
        compute_policy_loss(
            torch.zeros(1),
            torch.ones(1),
            eps_clip=0.2,
            eps_clip_high=0.2,
            policy_loss_type="missing",
        )
