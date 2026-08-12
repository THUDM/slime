"""CPU unit tests for ``slime.backends.megatron_utils.cp_utils.compute_mtp_losses``.

Regression test for https://github.com/THUDM/slime/issues/2131: multi-head
MTP models (``--mtp-num-layers > 1``) crashed at per-step logging because
the MTP-loss tracker's ``values`` tensor (one entry per MTP layer) was
squeezed through ``.item()``, which only works for a single-element tensor.

The CPU-only CI image does not ship megatron -- ``_cp_dist_helpers`` stubs
``megatron.core.mpu`` at import time so the subsequent ``cp_utils`` import
binds against the stub.
"""

from __future__ import annotations

# Import the helpers BEFORE the slime imports so the megatron stub lands
# in sys.modules first. pytest's prepend importmode puts this file's
# directory (``tests/``) on sys.path, which is what makes the bare-name
# import work without an ``__init__.py``.
import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils.cp_utils import compute_mtp_losses  # noqa: E402


NUM_GPUS = 0


@pytest.mark.unit
def test_single_mtp_layer_returns_one_loss():
    """``mtp_num_layers == 1`` keeps producing a single scaled loss value."""
    values = torch.tensor([2.0])
    assert compute_mtp_losses(values, mtp_loss_scale=0.5) == pytest.approx([1.0])


@pytest.mark.unit
def test_multi_mtp_layer_does_not_crash():
    """``mtp_num_layers > 1`` must not raise -- this is the reported crash:
    calling ``.item()`` on a multi-element tensor raises
    ``RuntimeError: a Tensor with 3 elements cannot be converted to Scalar``.
    """
    values = torch.tensor([2.0, 4.0, 6.0])
    losses = compute_mtp_losses(values, mtp_loss_scale=0.5)
    assert losses == pytest.approx([1.0, 2.0, 3.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
