"""CPU tests for ``off_policy_is_function`` (current-policy truncated IS).

The detached weight is ``clip(pi_theta / pi_rollout)`` using slime's TIS clip
args (``tis_clip_low`` / ``tis_clip``), matching ``vanilla_tis_function``.
Pure-torch (no megatron); ``NUM_GPUS = 0`` selects the CPU runner.
"""

import ast
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from slime.utils.ppo_utils import off_policy_is_function

NUM_GPUS = 0

LOSS_PATH = Path(__file__).parents[1] / "slime" / "backends" / "megatron_utils" / "loss.py"


@pytest.mark.unit
def test_off_policy_is_function_clips_weight_and_passes_masks_through():
    # ratio = exp(cur - rollout): 2.0 -> clamp 2.0 (tis_clip); 0.5 -> 0.8 (tis_clip_low); 1.0 unchanged
    cur = torch.tensor([1.0, 1.0, 1.0])
    rollout = cur - torch.tensor([2.0, 0.5, 1.0]).log()
    pg_loss = torch.tensor([1.0, 1.0, 1.0])
    loss_masks = [torch.ones(3)]
    args = Namespace(tis_clip_low=0.8, tis_clip=2.0)

    out_loss, out_masks, metrics = off_policy_is_function(
        args, pg_loss=pg_loss, cur_log_probs=[cur], rollout_log_probs=[rollout], loss_masks=loss_masks
    )

    expected_w = torch.tensor([2.0, 0.8, 1.0])
    assert torch.allclose(out_loss, pg_loss * expected_w)
    assert torch.allclose(metrics["tis_clipfrac"], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(metrics["tis"], torch.tensor([2.0, 0.5, 1.0]))
    assert out_masks is loss_masks


@pytest.mark.unit
def test_off_policy_is_stop_gradient_on_weights_with_reinforce_base():
    advantages = torch.tensor([2.0, -1.0, 0.5, 1.5])
    rollout = torch.tensor([-0.5, -0.2, -0.9, -0.3])
    log_probs = torch.tensor([-0.1, -0.4, -0.3, -0.8], requires_grad=True)
    args = Namespace(tis_clip_low=0.8, tis_clip=1.2)

    pg_loss = -advantages * log_probs
    pg_loss, _, _ = off_policy_is_function(
        args, pg_loss=pg_loss, cur_log_probs=[log_probs], rollout_log_probs=[rollout], loss_masks=[torch.ones(4)]
    )

    ratio = torch.exp(log_probs.detach() - rollout)
    clipped = ratio.clamp(args.tis_clip_low, args.tis_clip)
    assert torch.allclose(pg_loss, -clipped * advantages * log_probs.detach())

    pg_loss.sum().backward()
    assert torch.allclose(log_probs.grad, -clipped * advantages)


@pytest.mark.unit
def test_off_policy_is_default_tis_low_is_single_sided():
    # slime TIS default tis_clip_low=0 never clips ratios in (0, tis_clip].
    cur = torch.tensor([0.0, 0.0])
    rollout = cur - torch.tensor([10.0, 0.01]).log()  # ratios 10.0 (high) and ~0.01 (very low)
    pg_loss = torch.tensor([1.0, 1.0])
    args = Namespace(tis_clip_low=0.0, tis_clip=5.0)

    _, _, metrics = off_policy_is_function(
        args, pg_loss=pg_loss, cur_log_probs=[cur], rollout_log_probs=[rollout], loss_masks=[torch.ones(2)]
    )

    assert torch.allclose(metrics["tis_clipfrac"], torch.tensor([1.0, 0.0]))


def test_policy_loss_function_forwards_cur_log_probs_to_tis_hooks():
    module = ast.parse(LOSS_PATH.read_text())
    policy_loss_function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "policy_loss_function"
    )
    tis_kwargs_dicts = [
        node
        for node in ast.walk(policy_loss_function)
        if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == "train_log_probs" for key in node.keys)
    ]
    assert len(tis_kwargs_dicts) == 1
    keys = [key.value for key in tis_kwargs_dicts[0].keys if isinstance(key, ast.Constant)]
    assert "cur_log_probs" in keys


def test_off_policy_is_function_ignores_extra_tis_kwargs():
    args = Namespace(tis_clip_low=0.0, tis_clip=2.0)
    pg_loss = torch.tensor([1.0])
    cur = torch.tensor([0.0])
    rollout = torch.tensor([0.0])
    out_loss, masks, _ = off_policy_is_function(
        args,
        pg_loss=pg_loss,
        cur_log_probs=[cur],
        rollout_log_probs=[rollout],
        loss_masks=[torch.ones(1)],
        train_log_probs=[cur],
        total_lengths=[1],
        response_lengths=[1],
    )
    assert torch.allclose(out_loss, pg_loss)
    assert len(masks) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
