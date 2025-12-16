"""
Usage:
    PYTHONPATH=/root/Megatron-LM  python -m pytest tests/test_fsdp_mis.py -v
"""
from argparse import Namespace

import pytest
import torch

from examples.train_infer_mismatch_helper.mis_fsdp import compute_mis_weights_fsdp
from slime.backends.fsdp_utils.actor import vanilla_tis_function_fsdp


def create_mis_args(**overrides):
    defaults = {
        "use_tis": True,
        "tis_mode": "truncate",
        "tis_level": "token",
        "tis_upper_bound": 2.0,
        "tis_lower_bound": 0.5,
        "tis_batch_normalize": False,
        "use_rs": False,
        "rs_lower_bound": None,
        "rs_upper_bound": None,
        "rs_level": "token",
        "rs_veto_threshold": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.mark.parametrize(
    "use_tis,tis_clip,tis_clip_low",
    [
        (True, 2.0, 0.5),
        (True, 5.0, 0.1),
        (True, 1.5, 0.8),
    ],
)
def test_vanilla_tis_clipping(use_tis, tis_clip, tis_clip_low):
    args = Namespace(
        use_tis=use_tis,
        tis_clip=tis_clip,
        tis_clip_low=tis_clip_low,
    )

    train_log_probs = [
        torch.tensor([-1.0, -1.5, -2.0]),
        torch.tensor([-0.5, -1.0, -1.5, -2.0, -2.5]),
    ]
    rollout_log_probs = [
        torch.tensor([-2.0, -1.0, -1.5]),
        torch.tensor([-1.0, -0.5, -1.0, -1.5, -2.0]),
    ]
    loss_masks = [torch.ones(3), torch.ones(5)]
    pg_loss = torch.ones(8)

    pg_loss_out, masks_out, metrics = vanilla_tis_function_fsdp(
        args,
        pg_loss=pg_loss,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
    )

    raw_ratios = torch.exp(torch.cat(train_log_probs) - torch.cat(rollout_log_probs))
    expected_weights = raw_ratios.clamp(min=tis_clip_low, max=tis_clip)

    assert torch.allclose(pg_loss_out, expected_weights, atol=1e-5)
    assert torch.allclose(raw_ratios[0], torch.exp(torch.tensor(1.0)), atol=1e-5)
    assert torch.allclose(
        expected_weights[0], torch.tensor(min(tis_clip, torch.exp(torch.tensor(1.0)).item())), atol=1e-5
    )


@pytest.mark.parametrize(
    "tis_mode,tis_upper_bound,tis_lower_bound",
    [
        ("mask", 2.0, 0.5),
        ("truncate", 2.0, 0.5),
        ("clip", 2.0, 0.5),
    ],
)
def test_mis_modes(tis_mode, tis_upper_bound, tis_lower_bound):
    args = create_mis_args(
        tis_mode=tis_mode,
        tis_upper_bound=tis_upper_bound,
        tis_lower_bound=tis_lower_bound,
    )

    train_log_probs = [
        torch.tensor([-1.0, -1.5, -2.0]),
        torch.tensor([-0.1, -0.2, -0.3]),
    ]
    rollout_log_probs = [
        torch.tensor([-2.0, -1.0, -1.5]),
        torch.tensor([-5.0, -4.5, -4.0]),
    ]
    loss_masks = [torch.ones(3), torch.ones(3)]
    pg_loss = torch.ones(6)

    pg_loss_out, masks_out, metrics = compute_mis_weights_fsdp(
        args,
        pg_loss=pg_loss,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
    )

    raw_ratios = [torch.exp(t - r) for t, r in zip(train_log_probs, rollout_log_probs, strict=False)]

    if tis_mode == "mask":
        expected_masks = [((r >= tis_lower_bound) & (r <= tis_upper_bound)).float() for r in raw_ratios]
        for mask_out, expected_mask in zip(masks_out, expected_masks, strict=False):
            assert torch.equal(mask_out, expected_mask)
    elif tis_mode == "truncate":
        expected_weights = torch.cat([r.clamp(0, tis_upper_bound) for r in raw_ratios])
        assert torch.allclose(pg_loss_out, expected_weights, atol=1e-5)
    elif tis_mode == "clip":
        expected_weights = torch.cat([r.clamp(tis_lower_bound, tis_upper_bound) for r in raw_ratios])
        assert torch.allclose(pg_loss_out, expected_weights, atol=1e-5)


@pytest.mark.parametrize("batch_normalize", [True, False])
def test_mis_batch_normalization(batch_normalize):
    args = create_mis_args(tis_batch_normalize=batch_normalize)

    train_log_probs = [
        torch.tensor([-1.0, -1.5, -2.0]),
        torch.tensor([-0.5, -1.0, -1.5]),
    ]
    rollout_log_probs = [
        torch.tensor([-2.0, -1.0, -1.5]),
        torch.tensor([-1.0, -0.5, -1.0]),
    ]
    loss_masks = [torch.ones(3), torch.ones(3)]
    pg_loss = torch.ones(6)

    pg_loss_out, masks_out, metrics = compute_mis_weights_fsdp(
        args,
        pg_loss=pg_loss,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
    )

    raw_weights_flat = torch.exp(torch.cat(train_log_probs) - torch.cat(rollout_log_probs))
    clipped_weights = raw_weights_flat.clamp(0.5, 2.0)

    if batch_normalize:
        weights_mean = clipped_weights.mean()
        expected_weights = clipped_weights / weights_mean
        assert torch.allclose(pg_loss_out, expected_weights, atol=1e-5)
        assert torch.allclose(pg_loss_out.mean(), torch.tensor(1.0), atol=1e-5)
    else:
        assert torch.allclose(pg_loss_out, clipped_weights, atol=1e-5)


def test_mis_rejection_sampling():
    args = create_mis_args(
        use_rs=True,
        rs_lower_bound=0.5,
        rs_upper_bound=3.0,
        rs_veto_threshold=0.01,
        rs_level="token",
    )

    train_log_probs = [
        torch.tensor([-1.0, -1.5, -2.0]),
        torch.tensor([-10.0, -0.5, -1.0]),
    ]
    rollout_log_probs = [
        torch.tensor([-1.5, -1.2, -1.8]),
        torch.tensor([-5.0, -0.8, -0.9]),
    ]
    loss_masks = [torch.ones(3), torch.ones(3)]
    pg_loss = torch.ones(6)

    pg_loss_out, masks_out, metrics = compute_mis_weights_fsdp(
        args,
        pg_loss=pg_loss,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
    )

    raw_ratios = [torch.exp(t - r) for t, r in zip(train_log_probs, rollout_log_probs, strict=False)]

    assert torch.allclose(raw_ratios[1][0], torch.exp(torch.tensor(-5.0)), atol=1e-5)
    assert raw_ratios[1][0] < 0.01

    rs_mask_seq0 = ((raw_ratios[0] >= 0.5) & (raw_ratios[0] <= 3.0)).float()
    assert torch.equal(masks_out[0], rs_mask_seq0)
    assert torch.equal(masks_out[1], torch.zeros(3))


@pytest.mark.parametrize("tis_level", ["token", "sequence", "geometric"])
def test_mis_aggregation_levels(tis_level):
    args = create_mis_args(
        tis_level=tis_level,
        tis_mode="truncate",
    )

    train_log_probs = [
        torch.tensor([-1.0, -1.5, -2.0]),
        torch.tensor([-0.5, -1.0, -1.5]),
    ]
    rollout_log_probs = [
        torch.tensor([-2.0, -1.0, -1.5]),
        torch.tensor([-1.0, -0.5, -1.0]),
    ]
    loss_masks = [torch.ones(3), torch.ones(3)]
    pg_loss = torch.ones(6)

    pg_loss_out, masks_out, metrics = compute_mis_weights_fsdp(
        args,
        pg_loss=pg_loss,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
    )

    log_diffs = [t - r for t, r in zip(train_log_probs, rollout_log_probs, strict=False)]

    if tis_level == "token":
        expected_weights = torch.cat([torch.exp(ld).clamp(0, 2.0) for ld in log_diffs])
        assert torch.allclose(pg_loss_out, expected_weights, atol=1e-5)
    elif tis_level == "sequence":
        seq_weights = [torch.exp(ld.sum()).clamp(0, 2.0).expand_as(ld) for ld in log_diffs]
        expected_weights = torch.cat(seq_weights)
        assert torch.allclose(pg_loss_out, expected_weights, atol=1e-5)
        assert torch.allclose(pg_loss_out[0], pg_loss_out[1])
        assert torch.allclose(pg_loss_out[0], pg_loss_out[2])
    elif tis_level == "geometric":
        geo_weights = [torch.exp(ld.mean()).clamp(0, 2.0).expand_as(ld) for ld in log_diffs]
        expected_weights = torch.cat(geo_weights)
        assert torch.allclose(pg_loss_out, expected_weights, atol=1e-5)
        assert torch.allclose(pg_loss_out[3], pg_loss_out[4])
        assert torch.allclose(pg_loss_out[3], pg_loss_out[5])
