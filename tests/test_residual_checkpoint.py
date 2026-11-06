from __future__ import annotations

import pytest
import torch

from slime.backends.megatron_utils.alignment.residual_checkpoint import (
    EXACT_RESIDUAL_ATTRIBUTE,
    wrap_checkpoint,
)

NUM_GPUS = 0


def _direct_checkpoint(function, distribute_saved_activations, *args):
    assert distribute_saved_activations is False
    return function(*args)


def test_unaligned_checkpoint_is_unchanged():
    calls = []

    def original(function, distribute_saved_activations, *args):
        calls.append((function, distribute_saved_activations, args))
        return function(*args)

    hidden = torch.tensor([2.0], requires_grad=True)
    output = wrap_checkpoint(original)(lambda value: value * 3, False, hidden)

    assert output.item() == 6.0
    assert len(calls) == 1
    assert calls[0][1:] == (False, (hidden,))


def test_exact_residual_value_and_gradient_cross_checkpoint_boundary():
    hidden = torch.tensor([2.0], dtype=torch.bfloat16, requires_grad=True)
    exact_source = torch.tensor([2.25], dtype=torch.float32, requires_grad=True)
    setattr(hidden, EXACT_RESIDUAL_ATTRIBUTE, exact_source)

    def layer(value):
        exact = getattr(value, EXACT_RESIDUAL_ATTRIBUTE)
        return exact * 4

    output = wrap_checkpoint(_direct_checkpoint)(layer, False, hidden)
    output.sum().backward()

    torch.testing.assert_close(output, torch.tensor([9.0]))
    torch.testing.assert_close(hidden.grad, torch.tensor([4.0], dtype=torch.bfloat16))
    assert exact_source.grad is None
    assert getattr(hidden, EXACT_RESIDUAL_ATTRIBUTE) is exact_source


def test_temporary_residual_attribute_is_removed_after_recompute():
    hidden = torch.tensor([1.0], dtype=torch.bfloat16, requires_grad=True)
    exact = torch.tensor([1.125], dtype=torch.float32)
    setattr(hidden, EXACT_RESIDUAL_ATTRIBUTE, exact)

    seen = []

    def original(function, distribute_saved_activations, *args):
        recompute_hidden = args[0].detach().requires_grad_(True)
        seen.append(recompute_hidden)
        return function(recompute_hidden, *args[1:])

    output = wrap_checkpoint(original)(
        lambda value: getattr(value, EXACT_RESIDUAL_ATTRIBUTE) + 1,
        False,
        hidden,
    )

    torch.testing.assert_close(output, torch.tensor([2.125]))
    assert not hasattr(seen[0], EXACT_RESIDUAL_ATTRIBUTE)


@pytest.mark.parametrize(
    "exact",
    [
        torch.ones(2, dtype=torch.float32),
        torch.ones(1, dtype=torch.bfloat16),
    ],
)
def test_invalid_exact_residual_layout_fails_closed(exact):
    hidden = torch.ones(1, dtype=torch.bfloat16)
    setattr(hidden, EXACT_RESIDUAL_ATTRIBUTE, exact)

    with pytest.raises(RuntimeError, match="shape and use FP32"):
        wrap_checkpoint(_direct_checkpoint)(lambda value: value, False, hidden)


def test_distributed_saved_activations_fail_closed():
    hidden = torch.ones(1, dtype=torch.bfloat16)
    setattr(hidden, EXACT_RESIDUAL_ATTRIBUTE, torch.ones(1, dtype=torch.float32))

    with pytest.raises(RuntimeError, match="undistributed saved activations"):
        wrap_checkpoint(_direct_checkpoint)(lambda value: value, True, hidden)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
