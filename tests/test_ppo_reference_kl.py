"""Regression tests for preserving raw reference KL during PPO shaping."""

from __future__ import annotations

import torch

# Install the lightweight Megatron stub before importing the loss module.
import _cp_dist_helpers  # noqa: F401


def test_ppo_reward_shaping_does_not_mutate_reference_kl(monkeypatch) -> None:
    from megatron.core import mpu
    from slime.backends.megatron_utils import loss

    # ``compute_approx_kl`` is torch.compile'd in production. This regression
    # test targets the in-place mutation contract and should stay a fast CPU
    # unit test instead of compiling an Inductor kernel during collection.
    monkeypatch.setattr(
        loss,
        "compute_approx_kl",
        lambda log_probs, log_probs_base, kl_loss_type: log_probs - log_probs_base,
    )

    mpu.is_pipeline_last_stage = lambda: True
    mpu.get_context_parallel_rank = lambda: 0

    class Args:
        adaptive_kl_mode = "reward"
        advantage_estimator = "ppo"
        kl_coef = 0.1
        kl_loss_type = "k1"
        use_rollout_logprobs = False
        use_opd = False
        custom_advantage_function_path = None
        normalize_advantages = False
        gamma = 1.0
        lambd = 1.0

    raw_kl = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
    raw_before = raw_kl.clone()
    rollout_data = {
        "log_probs": [raw_kl.clone()],
        "ref_log_probs": [torch.zeros_like(raw_kl)],
        "rewards": [1.0],
        "values": [torch.zeros_like(raw_kl)],
        "response_lengths": [3],
        "total_lengths": [3],
        "loss_masks": [torch.ones_like(raw_kl)],
    }

    loss.compute_advantages_and_returns(Args(), rollout_data)

    torch.testing.assert_close(rollout_data["kl"][0], raw_before)
    # The task reward is added only to the shaped reward used for GAE.
    torch.testing.assert_close(
        rollout_data["returns"][0],
        torch.tensor([0.88, 0.90, 0.94]),
    )
