"""CPU tests for compute_reinforce_loss (plain ``-A * log pi_theta`` surrogate)."""

import pytest
import torch

from slime.utils.ppo_utils import compute_reinforce_loss

NUM_GPUS = 0


@pytest.mark.unit
def test_reinforce_loss_matches_closed_form():
    advantages = torch.tensor([2.0, -1.0, 0.5])
    log_probs = torch.tensor([-0.1, -0.2, -0.3])

    pg_loss, clipfrac = compute_reinforce_loss(advantages, log_probs)

    assert torch.allclose(pg_loss, -advantages * log_probs)
    assert torch.allclose(clipfrac, torch.zeros(3))


@pytest.mark.unit
def test_reinforce_gradient_flows_only_through_log_probs():
    advantages = torch.tensor([2.0, -1.0, 0.5])
    log_probs = torch.tensor([-0.1, -0.2, -0.3], requires_grad=True)

    pg_loss, _ = compute_reinforce_loss(advantages, log_probs)
    pg_loss.sum().backward()

    # d/d log_probs [ -A * log_probs ] = -A
    assert torch.allclose(log_probs.grad, -advantages)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
