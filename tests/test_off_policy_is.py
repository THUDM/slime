"""Unit tests for the ``off_policy_is_function`` importance-sampling correction
(slime/utils/ppo_utils.py).

It is truncated IS between the *current* policy and the *actual rollout generator*:
the (detached) weight is ``clip(pi_theta / pi_rollout)``. On a plain REINFORCE base
``-A * log pi`` it reproduces the CISPO surrogate (https://arxiv.org/abs/2506.13585).

Pure-torch (no megatron), like tests/test_chunked_gae.py; runs on CPU. NUM_GPUS = 0
selects the CPU runner in the changed-test CI matrix; the __main__ block lets CI run
it as a script. (The hook wiring in loss.py that supplies cur_log_probs imports
megatron and is exercised in the GPU CI suites.)
"""

from argparse import Namespace

import pytest
import torch

from slime.utils.ppo_utils import off_policy_is_function

# CPU-only test: selects the 0-GPU runner in the changed-test CI matrix.
NUM_GPUS = 0


@pytest.mark.unit
def test_off_policy_is_function_clips_weight_and_passes_masks_through():
    # ratio = exp(cur - rollout): ln(2) -> 2 -> clamp 1.2; ln(0.5) -> 0.5 -> 0.8; 0 -> 1.0
    cur = torch.tensor([1.0, 1.0, 1.0])
    rollout = cur - torch.tensor([2.0, 0.5, 1.0]).log()
    pg_loss = torch.tensor([1.0, 1.0, 1.0])
    loss_masks = [torch.ones(3)]
    args = Namespace(eps_clip=0.2, eps_clip_high=0.2)

    out_loss, out_masks, metrics = off_policy_is_function(
        args, pg_loss=pg_loss, cur_log_probs=[cur], rollout_log_probs=[rollout], loss_masks=loss_masks
    )

    expected_w = torch.tensor([1.2, 0.8, 1.0])
    assert torch.allclose(out_loss, pg_loss * expected_w)
    assert torch.allclose(metrics["is_clipfrac"], torch.tensor([1.0, 1.0, 0.0]))
    assert out_masks is loss_masks  # no rejection-sampling masking


@pytest.mark.unit
def test_off_policy_is_on_reinforce_base_equals_cispo_surrogate():
    # On a plain REINFORCE base (-A * log pi), off_policy_is_function reproduces the
    # CISPO surrogate exactly, with gradient flowing ONLY through log_probs.
    advantages = torch.tensor([2.0, -1.0, 0.5, 1.5])
    rollout = torch.tensor([-0.5, -0.2, -0.9, -0.3])  # behavior policy mu (frozen)
    log_probs = torch.tensor([-0.1, -0.4, -0.3, -0.8], requires_grad=True)
    args = Namespace(eps_clip=0.2, eps_clip_high=0.2)

    pg_loss = -advantages * log_probs  # plain REINFORCE base
    pg_loss, _, _ = off_policy_is_function(
        args, pg_loss=pg_loss, cur_log_probs=[log_probs], rollout_log_probs=[rollout], loss_masks=[torch.ones(4)]
    )

    ratio = torch.exp(log_probs.detach() - rollout)  # pi_theta / pi_rollout
    clipped = ratio.clamp(1 - args.eps_clip, 1 + args.eps_clip_high)
    assert torch.allclose(pg_loss, -clipped * advantages * log_probs.detach())

    pg_loss.sum().backward()
    # d/d log_probs [ -clip(ratio).detach() * A * log_probs ] = -clip(ratio) * A
    assert torch.allclose(log_probs.grad, -clipped * advantages)


@pytest.mark.unit
def test_off_policy_is_single_sided_when_eps_clip_one():
    # Canonical CISPO: eps_clip=1.0 disables the lower bound (ratio >= 0 never clipped low).
    cur = torch.tensor([0.0, 0.0])
    rollout = cur - torch.tensor([10.0, 0.01]).log()  # ratios 10.0 (high) and ~0.01 (very low)
    pg_loss = torch.tensor([1.0, 1.0])
    args = Namespace(eps_clip=1.0, eps_clip_high=4.0)

    _, _, metrics = off_policy_is_function(
        args, pg_loss=pg_loss, cur_log_probs=[cur], rollout_log_probs=[rollout], loss_masks=[torch.ones(2)]
    )

    # high ratio 10.0 > 1+eps_clip_high=5.0 clipped; low ratio ~0.01 >= 1-eps_clip=0.0 NOT clipped
    assert torch.allclose(metrics["is_clipfrac"], torch.tensor([1.0, 0.0]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
