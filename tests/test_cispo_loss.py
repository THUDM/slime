"""CPU tests for compute_cispo_loss (MiniMax-M1, https://arxiv.org/abs/2506.13585)."""

import math

import pytest
import torch

from slime.utils.ppo_utils import compute_cispo_loss

NUM_GPUS = 0

ADVANTAGES = torch.tensor([1.0, -0.5, 2.0, -1.0])
LOG_PROBS = torch.tensor([-0.7, -1.2, -0.4, -2.1])

# (eps_clip, eps_clip_high, raw IS ratios, ratios after clamp to [1 - eps_clip, 1 + eps_clip_high])
CLIP_CASES = [
    pytest.param(0.2, 0.28, [1.0, 1.14, 1.56, 0.4], [1.0, 1.14, 1.28, 0.8], id="ppo_band"),
    pytest.param(1.0, 4.0, [1.0, 3.0, 9.0, 0.4], [1.0, 3.0, 5.0, 0.4], id="wide_minimax_band"),
]


@pytest.mark.parametrize("eps_clip, eps_clip_high, ratios, clamped", CLIP_CASES)
def test_compute_cispo_loss_matches_closed_form_surrogate(eps_clip, eps_clip_high, ratios, clamped):
    ppo_kl = -torch.tensor([math.log(r) for r in ratios])

    pg_losses, clipfrac = compute_cispo_loss(ppo_kl, LOG_PROBS, ADVANTAGES, eps_clip, eps_clip_high)

    expected_losses = -torch.tensor(clamped) * ADVANTAGES * LOG_PROBS
    torch.testing.assert_close(pg_losses, expected_losses, rtol=1e-6, atol=1e-6)
    expected_clipfrac = torch.tensor([float(c != r) for c, r in zip(clamped, ratios, strict=True)])
    torch.testing.assert_close(clipfrac, expected_clipfrac)


@pytest.mark.parametrize("eps_clip, eps_clip_high, ratios, clamped", CLIP_CASES)
def test_compute_cispo_loss_gradient_flows_only_through_log_probs(eps_clip, eps_clip_high, ratios, clamped):
    # ratio = exp(-ppo_kl) = exp(log_ratios): if CISPO failed to stop-gradient the
    # clipped IS ratio, backward would populate log_ratios.grad.
    log_ratios = torch.tensor([math.log(r) for r in ratios], requires_grad=True)
    ppo_kl = -log_ratios
    log_probs = LOG_PROBS.clone().requires_grad_()

    pg_losses, _ = compute_cispo_loss(ppo_kl, log_probs, ADVANTAGES, eps_clip, eps_clip_high)
    pg_losses.sum().backward()

    torch.testing.assert_close(log_probs.grad, -torch.tensor(clamped) * ADVANTAGES, rtol=1e-6, atol=1e-6)
    assert log_ratios.grad is None or torch.all(
        log_ratios.grad == 0
    ), f"CISPO must stop-gradient on the IS ratio; log_ratios.grad={log_ratios.grad}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
