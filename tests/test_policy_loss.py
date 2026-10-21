"""CPU tests for PPO policy-loss clipping and its training-path wiring."""

import ast
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("scalar_metrics", [False, True])
def test_training_loss_keeps_metric_values_normalization_and_gradients(per_token, dtype, scalar_metrics):
    # Exercise the actual wrapper without importing the optional Megatron stack.
    path = Path(__file__).parents[1] / "slime" / "backends" / "megatron_utils" / "loss.py"
    source = path.read_text()
    function = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "loss_function"
    )
    logits = torch.tensor([1.5, 2.0], dtype=dtype, requires_grad=True)
    metrics = {"loss": logits.square().mean(), "one_element": logits[:1]}
    if scalar_metrics:
        metrics.update(count=3, fraction=0.25)
    args = SimpleNamespace(
        loss_type="policy_loss",
        calculate_per_token_loss=per_token,
        recompute_loss_function=False,
        allgather_cp=False,
    )
    batch = {
        "loss_masks": [torch.tensor([0.0, 1.0, 1.0])],
        "total_lengths": [5],
        "response_lengths": [3],
        "rollout_mask_sums": [2],
    }
    namespace = {
        "torch": torch,
        "mpu": SimpleNamespace(get_data_parallel_world_size=lambda **_: 2, get_context_parallel_world_size=lambda: 2),
        "get_sum_of_sample_mean": lambda *args: None,
        "policy_loss_function": lambda *args: (logits.square().mean(), metrics),
    }
    exec(
        compile("from __future__ import annotations\n" + ast.get_source_segment(source, function), str(path), "exec"),
        namespace,
    )
    loss, normalizer, report = namespace["loss_function"](args, batch, 4, 128, logits)
    # The former construction defines the supported scalar shapes and dtypes.
    expected = torch.tensor(
        [torch.tensor(2.0) if per_token else 0]
        + [value.detach() if isinstance(value, torch.Tensor) else value for value in metrics.values()]
    )
    torch.testing.assert_close(report["values"], expected)
    assert report["keys"] == list(metrics)
    assert not report["values"].requires_grad
    assert report["values"].grad_fn is None
    torch.testing.assert_close(normalizer, torch.tensor(2.0) if per_token else torch.tensor(1))
    loss.backward()
    scale = 2 if per_token else 4 / 128 * 2
    torch.testing.assert_close(logits.grad, logits.detach() * scale)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
