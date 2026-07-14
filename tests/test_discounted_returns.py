import sys
import types

import pytest
import torch

from slime.utils.ppo_utils import chunked_discounted_returns, chunked_gae, get_reinforce_plus_plus_returns, vanilla_gae


NUM_GPUS = 0


def _serial_discounted_returns(rewards: torch.Tensor, discount: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running_return = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(rewards.size(1))):
        running_return = rewards[:, t] + discount * running_return
        returns[:, t] = running_return
    return returns


@pytest.mark.parametrize("discount", [0.0, 0.5, 0.99, 1.0])
@pytest.mark.parametrize("batch_size,sequence_length", [(1, 1), (3, 127), (3, 128), (3, 129), (3, 1000)])
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float32, 1e-4, 1e-5),
        (torch.float64, 1e-10, 1e-10),
    ],
)
def test_chunked_discounted_returns_matches_serial(discount, batch_size, sequence_length, dtype, atol, rtol):
    torch.manual_seed(0)
    rewards = torch.randn(batch_size, sequence_length, dtype=dtype)

    expected = _serial_discounted_returns(rewards, discount)
    actual = chunked_discounted_returns(rewards, discount)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert actual.dtype == rewards.dtype
    assert actual.device == rewards.device


def test_chunked_discounted_returns_preserves_right_padding():
    torch.manual_seed(0)
    lengths = [3, 129, 511]
    rewards = torch.zeros(len(lengths), max(lengths))
    for i, length in enumerate(lengths):
        rewards[i, :length] = torch.randn(length)

    actual = chunked_discounted_returns(rewards, 0.99)

    for i, length in enumerate(lengths):
        expected = _serial_discounted_returns(rewards[i : i + 1, :length], 0.99)[0]
        torch.testing.assert_close(actual[i, :length], expected, atol=1e-4, rtol=1e-5)
        assert torch.count_nonzero(actual[i, length:]) == 0


def test_reinforce_plus_plus_returns_matches_serial_for_variable_lengths(monkeypatch):
    mpu = types.SimpleNamespace(get_context_parallel_world_size=lambda: 1)
    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    megatron_core.mpu = mpu
    megatron.core = megatron_core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)

    rewards = torch.tensor([2.0, -1.0])
    kl = [torch.tensor([0.2, 0.1, 0.3, 0.4, 0.5]), torch.tensor([0.4, 0.2, 0.1])]
    loss_masks = [torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]), torch.tensor([1.0, 1.0, 1.0])]
    kl_coef = 0.1
    gamma = 0.99

    actual = get_reinforce_plus_plus_returns(
        rewards=rewards,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=[3, 3],
        total_lengths=[5, 3],
        kl_coef=kl_coef,
        gamma=gamma,
    )

    expected = []
    for reward, kl_for_seq, mask in zip(rewards, kl, loss_masks, strict=True):
        token_rewards = -kl_coef * kl_for_seq * mask
        token_rewards[mask.nonzero(as_tuple=True)[0][-1]] += reward
        expected.append(_serial_discounted_returns(token_rewards.unsqueeze(0), gamma)[0])

    assert len(actual) == len(expected)
    for actual_for_seq, expected_for_seq in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_for_seq, expected_for_seq, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("sequence_length", [127, 128, 129])
def test_chunked_gae_matches_serial_after_scan_reuse(sequence_length):
    torch.manual_seed(0)
    rewards = torch.randn(3, sequence_length)
    values = torch.randn(3, sequence_length)

    expected = vanilla_gae(rewards, values, gamma=0.99, lambd=0.95)
    actual = chunked_gae(rewards, values, gamma=0.99, lambd=0.95)

    torch.testing.assert_close(actual[0], expected[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
