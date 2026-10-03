import sys
import types
from argparse import Namespace
from datetime import timedelta

import pytest
import torch

from slime.utils.ppo_utils import compute_approx_kl

NUM_GPUS = 0


def _terminal_reward_worker(rank, world_size, rendezvous):
    import torch.distributed as dist

    from _cp_dist_helpers import stub_megatron_in_worker

    stub_megatron_in_worker(world_size, rank)
    from megatron.core import mpu

    mpu.is_pipeline_last_stage = lambda: True
    mpu.get_context_parallel_group = lambda: dist.group.WORLD
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp
        from slime.backends.megatron_utils.loss import compute_advantages_and_returns

        args = Namespace(
            advantage_estimator="ppo",
            kl_coef=0.2,
            kl_loss_type="k1",
            use_rollout_logprobs=False,
            custom_advantage_function_path=None,
            normalize_advantages=False,
            use_opd=False,
            gamma=0.97,
            lambd=0.9,
        )
        # Includes aligned lengths, a terminal token outside CP rank zero,
        # and a rank-zero shard that contains no response tokens at all.
        for total_length, response_length in [(16, 9), (8, 5), (9, 3), (9, 8), (7, 1)]:
            values = torch.arange(response_length, dtype=torch.float32) * 0.1
            log_probs = -0.3 - values
            ref_log_probs = torch.full_like(values, -0.5)
            reward = 1.5

            def local(tensor, total_length=total_length, response_length=response_length):
                return slice_log_prob_with_cp(tensor, total_length, response_length)

            data = {
                "log_probs": [local(log_probs)],
                "ref_log_probs": [local(ref_log_probs)],
                "values": [local(values)],
                "rewards": [reward],
                "total_lengths": [total_length],
                "response_lengths": [response_length],
                "loss_masks": [torch.ones(response_length)],
            }
            compute_advantages_and_returns(args, data)

            rewards = -args.kl_coef * (log_probs - ref_log_probs)
            rewards[-1] += reward
            expected = torch.zeros_like(values)
            running = 0.0
            for t in reversed(range(response_length)):
                next_value = values[t + 1] if t + 1 < response_length else 0.0
                delta = rewards[t] + args.gamma * next_value - values[t]
                running = delta + args.gamma * args.lambd * running
                expected[t] = running
            torch.testing.assert_close(data["advantages"][0], local(expected))
            torch.testing.assert_close(data["returns"][0], local(expected + values))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_ppo_terminal_reward_matches_unsharded_gae(cp_size, tmp_path):
    torch.multiprocessing.spawn(
        _terminal_reward_worker,
        args=(cp_size, (tmp_path / "rendezvous").as_uri()),
        nprocs=cp_size,
    )


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
