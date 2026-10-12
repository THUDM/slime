from argparse import Namespace

import torch

from slime.utils.ppo_utils import compute_opsm_mask


def test_opsm_rejects_entire_high_kl_negative_sequence_with_mixed_token_advantages():
    args = Namespace(opsm_delta=0.1)

    opsm_mask, opsm_clipfrac = compute_opsm_mask(
        args=args,
        full_log_probs=[torch.tensor([0.0, 0.0, 0.0])],
        full_old_log_probs=[torch.tensor([1.0, 1.0, 1.0])],
        full_advantages=[torch.tensor([-0.5, 0.25, -0.25])],
        loss_masks=[torch.tensor([1.0, 1.0, 1.0])],
    )

    assert torch.equal(opsm_mask, torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(opsm_clipfrac, torch.tensor(1.0))


def test_opsm_keeps_low_kl_sequence_active():
    args = Namespace(opsm_delta=0.1)

    opsm_mask, opsm_clipfrac = compute_opsm_mask(
        args=args,
        full_log_probs=[torch.tensor([0.0, 0.0, 0.0])],
        full_old_log_probs=[torch.tensor([0.01, 0.01, 0.01])],
        full_advantages=[torch.tensor([-0.5, 0.25, -0.25])],
        loss_masks=[torch.tensor([1.0, 1.0, 1.0])],
    )

    assert torch.equal(opsm_mask, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(opsm_clipfrac, torch.tensor(0.0))


def test_opsm_returns_local_mask_when_sequence_inputs_are_full_length():
    args = Namespace(opsm_delta=0.1)

    opsm_mask, opsm_clipfrac = compute_opsm_mask(
        args=args,
        full_log_probs=[torch.tensor([0.0, 0.0, 0.0, 0.0])],
        full_old_log_probs=[torch.tensor([1.0, 1.0, 1.0, 1.0])],
        full_advantages=[torch.tensor([-0.5, 0.25, -0.25, -0.5])],
        loss_masks=[torch.tensor([1.0, 1.0, 1.0, 1.0])],
        local_advantages=[torch.tensor([0.25, -0.5])],
    )

    assert torch.equal(opsm_mask, torch.tensor([0.0, 0.0]))
    assert torch.equal(opsm_clipfrac, torch.tensor(1.0))
