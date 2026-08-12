"""CPU tests for PPO policy-loss clipping and its training-path wiring."""

import ast
from pathlib import Path

import pytest
import torch

from slime.utils.ppo_utils import compute_policy_loss

NUM_GPUS = 0


def test_compute_policy_loss_applies_dual_clip_to_negative_advantages():
    ratios = torch.tensor([2.0, 2.0, 0.5])
    ppo_kl = -ratios.log()
    advantages = torch.tensor([2.0, -2.0, 2.0])

    losses, _ = compute_policy_loss(
        ppo_kl,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=1.5,
    )

    torch.testing.assert_close(losses, torch.tensor([-2.4, 3.0, -1.0]))


def test_policy_loss_function_forwards_eps_clip_c():
    loss_path = Path(__file__).parents[1] / "slime" / "backends" / "megatron_utils" / "loss.py"
    module = ast.parse(loss_path.read_text())
    policy_loss_function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "policy_loss_function"
    )
    compute_policy_loss_calls = [
        node
        for node in ast.walk(policy_loss_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "compute_policy_loss"
    ]

    assert len(compute_policy_loss_calls) == 1
    eps_clip_c_keyword = next(
        (keyword for keyword in compute_policy_loss_calls[0].keywords if keyword.arg == "eps_clip_c"),
        None,
    )
    assert eps_clip_c_keyword is not None
    assert ast.dump(eps_clip_c_keyword.value) == ast.dump(
        ast.Attribute(value=ast.Name(id="args", ctx=ast.Load()), attr="eps_clip_c", ctx=ast.Load())
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
