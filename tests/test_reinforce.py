"""CPU tests for compute_reinforce_loss (plain ``-A * log pi_theta`` surrogate)."""

import ast
from pathlib import Path

import pytest
import torch

from slime.utils.ppo_utils import compute_reinforce_loss

NUM_GPUS = 0

LOSS_PATH = Path(__file__).parents[1] / "slime" / "backends" / "megatron_utils" / "loss.py"


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


def _function_named(module: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_policy_loss_function_dispatches_reinforce_estimator():
    module = ast.parse(LOSS_PATH.read_text())
    policy_loss_function = _function_named(module, "policy_loss_function")
    compute_reinforce_loss_calls = [
        node
        for node in ast.walk(policy_loss_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "compute_reinforce_loss"
    ]
    assert len(compute_reinforce_loss_calls) == 1

    advantages_fn = _function_named(module, "compute_advantages_and_returns")
    estimator_lists = [
        [elt.value for elt in node.elts if isinstance(elt, ast.Constant)]
        for node in ast.walk(advantages_fn)
        if isinstance(node, ast.List)
    ]
    assert any("reinforce" in values and "grpo" in values for values in estimator_lists)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
