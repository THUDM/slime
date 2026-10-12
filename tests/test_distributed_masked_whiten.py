"""CPU tests for distributed_masked_whiten shift_mean paths."""

import pytest
import torch
import torch.distributed as dist

from slime.utils.distributed_utils import distributed_masked_whiten

NUM_GPUS = 0
EPSILON = 1e-8


def _noop_all_reduce(tensor, group=None, op=None, **kwargs):
    """Single-rank stand-in: leave the local stats tensor unchanged."""
    return tensor


def _masked_mean_var(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Global mean / Bessel-corrected var matching distributed_masked_whiten."""
    mask_sum = mask.sum()
    global_mean = (values * mask).sum() / mask_sum
    global_mean_sq = ((values**2) * mask).sum() / mask_sum
    global_var = global_mean_sq - global_mean**2
    if mask_sum.item() >= 2:
        global_var = global_var * (mask_sum / (mask_sum - 1))
    return global_mean, global_var


@pytest.fixture
def values_and_mask():
    # Non-zero mean so the two shift_mean paths are distinguishable.
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, -1.0], dtype=torch.float32)
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    return values, mask


def test_shift_mean_true_is_zero_mean_and_matches_current_formula(monkeypatch, values_and_mask):
    monkeypatch.setattr(dist, "all_reduce", _noop_all_reduce)
    values, mask = values_and_mask
    mean, var = _masked_mean_var(values, mask)

    out = distributed_masked_whiten(values, mask, shift_mean=True, epsilon=EPSILON)

    expected = (values - mean) * torch.rsqrt(var + EPSILON)
    torch.testing.assert_close(out, expected)
    masked_out_mean = (out * mask).sum() / mask.sum()
    assert masked_out_mean.abs().item() < 1e-5


def test_shift_mean_false_scales_only_without_mean_add_back(monkeypatch, values_and_mask):
    monkeypatch.setattr(dist, "all_reduce", _noop_all_reduce)
    values, mask = values_and_mask
    mean, var = _masked_mean_var(values, mask)

    out = distributed_masked_whiten(values, mask, shift_mean=False, epsilon=EPSILON)

    expected = values * torch.rsqrt(var + EPSILON)
    torch.testing.assert_close(out, expected)

    old_add_back = (values - mean) * torch.rsqrt(var + EPSILON) + mean
    assert not torch.allclose(out, old_add_back)

    orig_mean = (values * mask).sum() / mask.sum()
    out_mean = (out * mask).sum() / mask.sum()
    assert not torch.allclose(out_mean, orig_mean)


def test_both_paths_share_bessel_corrected_var(monkeypatch, values_and_mask):
    monkeypatch.setattr(dist, "all_reduce", _noop_all_reduce)
    values, mask = values_and_mask
    mean, var = _masked_mean_var(values, mask)

    pop_var = ((values**2) * mask).sum() / mask.sum() - mean**2
    assert var > pop_var

    scale = torch.rsqrt(var + EPSILON)
    pop_scale = torch.rsqrt(pop_var + EPSILON)
    assert not torch.allclose(scale, pop_scale)

    out_shift = distributed_masked_whiten(values, mask, shift_mean=True, epsilon=EPSILON)
    out_noshift = distributed_masked_whiten(values, mask, shift_mean=False, epsilon=EPSILON)

    torch.testing.assert_close(out_shift, (values - mean) * scale)
    torch.testing.assert_close(out_noshift, values * scale)
    assert not torch.allclose(out_shift, (values - mean) * pop_scale)
    assert not torch.allclose(out_noshift, values * pop_scale)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
