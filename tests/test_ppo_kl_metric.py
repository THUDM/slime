import sys
import types
from argparse import Namespace

import pytest
import torch

NUM_GPUS = 0


@pytest.mark.unit
def test_ppo_estimator_does_not_corrupt_logged_kl(monkeypatch):
    """The ppo branch turns kl into a reward (-kl_coef * kl, plus the scalar
    reward at the last token). It must do so out-of-place: rollout_data["kl"] is
    the source for the logged rollout/kl metric, and every other estimator treats
    kl as read-only. An in-place update overwrites the logged KL with the reward.
    """
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
        from slime.utils.ppo_utils import compute_approx_kl

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
