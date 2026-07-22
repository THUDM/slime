"""Unit tests for the segment-gate and safe-region-align helpers.

These are pure-torch tests (no Megatron / no cluster). They are skipped when
torch is unavailable.
"""

import pytest

torch = pytest.importorskip("torch")

from slime.utils.ppo_utils import compute_segment_pg_weight, compute_rollout_align_loss  # noqa: E402

SEG_DEFAULTS = dict(
    neg_delta_threshold=-0.5,
    neg_adv_max=0.0,
    neg_weight=0.3,
    severe_delta_threshold=-1.5,
    bad_delta_threshold=-6.0,
    bad_fraction_threshold=0.02,
    severe_weight=0.1,
)


def test_segment_gate_neg_severe_and_normal():
    # Sample 1: seg0 = negative (mean -1.0), seg1 = severe (mean -2.0), both adv<0.
    raw_delta_s1 = torch.tensor([-1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0])
    adv_s1 = torch.full((8,), -0.5)
    mask_s1 = torch.ones(8)
    # Sample 2: one normal segment (delta 0, adv +1).
    raw_delta_s2 = torch.zeros(4)
    adv_s2 = torch.ones(4)
    mask_s2 = torch.ones(4)

    seg_weight, metrics = compute_segment_pg_weight(
        raw_delta_list=[raw_delta_s1, raw_delta_s2],
        advantages_list=[adv_s1, adv_s2],
        loss_masks=[mask_s1, mask_s2],
        segment_size=4,
        **SEG_DEFAULTS,
    )

    expected = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0])
    assert torch.allclose(seg_weight, expected)
    assert pytest.approx(metrics["seg_neg_frac"].item()) == 1 / 3
    assert pytest.approx(metrics["seg_severe_frac"].item()) == 1 / 3


def test_segment_gate_bad_fraction_triggers_severe():
    # 10-token segment: mean = -0.7 (negative, NOT below severe -1.5), but one token
    # at -7 gives bad_fraction 0.1 > 0.02 -> severe via bad-fraction trigger.
    raw_delta = torch.tensor([0.0] * 9 + [-7.0])
    adv = torch.full((10,), -1.0)
    mask = torch.ones(10)

    seg_weight, metrics = compute_segment_pg_weight(
        raw_delta_list=[raw_delta],
        advantages_list=[adv],
        loss_masks=[mask],
        segment_size=10,
        **SEG_DEFAULTS,
    )

    assert torch.allclose(seg_weight, torch.full((10,), 0.1))
    assert pytest.approx(metrics["seg_severe_frac"].item()) == 1.0
    assert pytest.approx(metrics["seg_neg_frac"].item()) == 0.0


def test_segment_gate_positive_adv_not_gated():
    # delta is very negative but adv > 0 -> not gated (weight stays 1).
    raw_delta = torch.full((4,), -3.0)
    adv = torch.ones(4)
    mask = torch.ones(4)
    seg_weight, metrics = compute_segment_pg_weight(
        raw_delta_list=[raw_delta],
        advantages_list=[adv],
        loss_masks=[mask],
        segment_size=4,
        **SEG_DEFAULTS,
    )
    assert torch.allclose(seg_weight, torch.ones(4))
    assert metrics["seg_neg_frac"].item() == 0.0
    assert metrics["seg_severe_frac"].item() == 0.0


def test_segment_gate_fully_masked_segment_excluded():
    # First segment fully masked -> weight 1.0 and excluded from fractions.
    # Second segment negative and gated.
    raw_delta = torch.tensor([-2.0, -2.0, -1.0, -1.0])
    adv = torch.full((4,), -0.5)
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0])
    seg_weight, metrics = compute_segment_pg_weight(
        raw_delta_list=[raw_delta],
        advantages_list=[adv],
        loss_masks=[mask],
        segment_size=2,
        **SEG_DEFAULTS,
    )
    # seg0 (masked) -> 1.0 ; seg1 (neg) -> 0.3
    assert torch.allclose(seg_weight, torch.tensor([1.0, 1.0, 0.3, 0.3]))
    # only one valid segment, and it is negative
    assert pytest.approx(metrics["seg_neg_frac"].item()) == 1.0


def _single_sample_mean(mask):
    def fn(x):
        return (x * mask).sum() / torch.clamp_min(mask.sum(), 1)

    return fn


class _Args:
    rollout_align_huber_beta = 1.0
    rollout_align_delta_min = -6.0
    rollout_align_delta_max = -0.5
    rollout_align_adv_min = 0.0
    rollout_align_adv_max = None


def test_align_window_selection():
    # token0: d=-0.3 (above -0.5) -> out
    # token1: d=-1.0 in [-6,-0.5], adv>0 -> IN
    # token2: d=-7.0 below -6 -> out
    # token3: d=-1.0 in window but adv<=0 -> out
    raw_delta_grad = torch.tensor([-0.3, -1.0, -7.0, -1.0], requires_grad=True)
    adv = torch.tensor([1.0, 1.0, 1.0, -0.5])
    mask = torch.ones(4)

    align_loss, align_frac = compute_rollout_align_loss(
        raw_delta_grad=raw_delta_grad,
        advantages_detached=adv,
        args=_Args(),
        sum_of_sample_mean=_single_sample_mean(mask),
    )

    # only token1 in window; huber(-1.0, beta=1) = |x| - 0.5 = 0.5
    assert pytest.approx(align_frac.item()) == 1 / 4
    assert pytest.approx(align_loss.item(), abs=1e-6) == 0.5 / 4

    # gradient flows to the in-window token
    align_loss.backward()
    grad = raw_delta_grad.grad
    assert grad[1].item() != 0.0
    assert grad[0].item() == 0.0
    assert grad[2].item() == 0.0
    assert grad[3].item() == 0.0


def test_align_adv_max_bound():
    args = _Args()
    args.rollout_align_adv_max = 0.5  # exclude adv >= 0.5
    raw_delta_grad = torch.tensor([-1.0, -1.0])
    adv = torch.tensor([0.2, 1.0])  # first inside (<0.5), second excluded
    mask = torch.ones(2)
    align_loss, align_frac = compute_rollout_align_loss(
        raw_delta_grad=raw_delta_grad,
        advantages_detached=adv,
        args=args,
        sum_of_sample_mean=_single_sample_mean(mask),
    )
    assert pytest.approx(align_frac.item()) == 1 / 2
