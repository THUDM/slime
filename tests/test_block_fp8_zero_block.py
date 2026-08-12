"""CPU unit tests for block-wise FP8 quantization in tools/convert_hf_to_fp8.py.

Pins the contract that an all-zero quantization block must not produce
NaN weights. ``block_fp8`` (the default quantization strategy) computed
``scale = block_max / FP8_MAX`` without clamping the block max away from
zero, unlike ``channel_fp8`` and ``tensor_fp8`` which both clamp to
1e-12. An all-zero block (padding rows, an unused MoE expert, ...) gave
``scale == 0`` and ``qweight == 0 / 0 == NaN``, silently writing NaN
weights into the converted checkpoint.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("safetensors")
torch = pytest.importorskip("torch")

NUM_GPUS = 0


def _load_converter():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "convert_hf_to_fp8.py"
    spec = importlib.util.spec_from_file_location("convert_hf_to_fp8", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def converter():
    return _load_converter()


@pytest.mark.unit
def test_block_fp8_all_zero_block_has_no_nan(converter):
    # First 128x128 tile is non-zero, the other three are all-zero blocks.
    weight = torch.zeros(256, 256, dtype=torch.bfloat16)
    weight[0, 0] = 1.0

    qweight, scale = converter.block_fp8(weight, (128, 128))

    assert not torch.isnan(qweight.float()).any()
    assert not torch.isinf(qweight.float()).any()
    # No scale may be zero: dequantization multiplies by the scale, and a
    # zero scale is exactly what turned the all-zero block into NaN.
    assert (scale > 0).all()


@pytest.mark.unit
def test_block_fp8_zero_block_roundtrips_to_zero(converter):
    weight = torch.zeros(128, 128, dtype=torch.bfloat16)

    qweight, scale = converter.block_fp8(weight, (128, 128))

    dequantized = qweight.float() * scale.float()
    assert (dequantized == 0).all()


@pytest.mark.unit
def test_block_fp8_nonzero_blocks_unaffected(converter):
    torch.manual_seed(0)
    weight = torch.randn(256, 256, dtype=torch.float32).to(torch.bfloat16)

    qweight, scale = converter.block_fp8(weight, (128, 128))

    assert not torch.isnan(qweight.float()).any()
    dequantized = qweight.float() * scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1).float()
    max_err = (dequantized - weight.float()).abs().max()
    # FP8 e4m3 relative error is ~2^-3, so a tolerance of 0.5 is generous
    # for randn-scale values and only guards against gross corruption.
    assert max_err < 0.5
