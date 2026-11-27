"""Equivalence tests against a REAL HuggingFace VL processor.

These prove the strongest correctness claim for the async path: given identical
inputs, ``_prepare_prompt_ids_async`` returns the same ``prompt_ids`` and writes
the same ``multimodal_train_inputs`` tensors (``pixel_values``,
``image_grid_thw``, ...) as the original synchronous ``_prepare_prompt_ids`` --
byte-for-byte / ``torch.equal`` -- and that this still holds when many samples
are processed concurrently through the shared processor (a real-world shakeout
for HF fast-tokenizer / processor thread-safety).

Run with a local VL checkpoint, e.g.::

    SLIME_TEST_VL_CKPT=/root/models/Qwen2.5-VL-3B-Instruct \
        pytest tests/test_async_multimodal_real_processor.py -v

The test auto-skips when transformers is unavailable or the env var is unset, so
it is safe in CPU-only CI.
"""

from __future__ import annotations

import asyncio
import copy
import os

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from slime.rollout import sglang_rollout  # noqa: E402
from slime.utils import processing_utils  # noqa: E402
from slime.utils.processing_utils import (  # noqa: E402
    async_encode_image_for_rollout_engine,
    encode_image_for_rollout_engine,
    process_vision_info,
)
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0

CKPT = os.environ.get("SLIME_TEST_VL_CKPT")

pytestmark = pytest.mark.skipif(
    not CKPT,
    reason="Set SLIME_TEST_VL_CKPT=/path/to/vl_checkpoint to run real-processor equivalence tests.",
)


def _assert_deep_equal(a, b, path="root"):
    """Recursively assert equality, handling tensors / ndarrays / containers."""
    assert type(a) is type(b) or (a is None) == (b is None), f"{path}: type {type(a)} != {type(b)}"
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{path}: shape {tuple(a.shape)} != {tuple(b.shape)}"
        assert torch.equal(a, b), f"{path}: tensor values differ"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys {set(a)} != {set(b)}"
        for k in a:
            _assert_deep_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_deep_equal(x, y, f"{path}[{i}]")
    else:
        try:
            import numpy as np

            if isinstance(a, np.ndarray):
                assert np.array_equal(a, b), f"{path}: ndarray values differ"
                return
        except ImportError:
            pass
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture(scope="module")
def loaded():
    tokenizer = processing_utils.load_tokenizer(CKPT, trust_remote_code=True)
    processor = processing_utils.load_processor(CKPT, trust_remote_code=True)
    assert processor is not None, f"{CKPT} did not yield a usable processor"
    return tokenizer, processor


def _make_sample(tokenizer, processor, *, color, size=(64, 64), text="Describe the picture."):
    """Build a Sample exactly the way slime.utils.data.Dataset builds a VL row."""
    image = Image.new("RGB", size, color=color)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    multimodal_inputs = process_vision_info(messages, processor)
    return Sample(prompt=prompt_str, multimodal_inputs=multimodal_inputs)


@pytest.mark.integration
def test_real_processor_prompt_ids_and_train_inputs_match(loaded):
    tokenizer, processor = loaded
    base = _make_sample(tokenizer, processor, color=(200, 30, 30))

    sample_sync = copy.deepcopy(base)
    sample_async = copy.deepcopy(base)

    ids_sync = sglang_rollout._prepare_prompt_ids(sample_sync, tokenizer, processor)
    ids_async = asyncio.run(sglang_rollout._prepare_prompt_ids_async(sample_async, tokenizer, processor))

    assert ids_sync == ids_async, "prompt_ids from async path differ from sync path"
    _assert_deep_equal(
        sample_sync.multimodal_train_inputs,
        sample_async.multimodal_train_inputs,
        "multimodal_train_inputs",
    )


@pytest.mark.integration
def test_real_image_encoding_is_byte_identical(loaded):
    tokenizer, processor = loaded
    sample = _make_sample(tokenizer, processor, color=(10, 120, 240))
    images = sample.multimodal_inputs["images"]
    assert images, "expected at least one image from process_vision_info"

    sync_encoded = [encode_image_for_rollout_engine(im) for im in images]

    async def gather():
        return await asyncio.gather(*(async_encode_image_for_rollout_engine(im) for im in images))

    async_encoded = asyncio.run(gather())
    assert sync_encoded == async_encoded


@pytest.mark.integration
def test_real_processor_is_consistent_under_concurrency(loaded):
    """Run many distinct samples through the async path concurrently, sharing one
    processor, and compare each result against its own synchronous reference.

    This is the real thread-safety shakeout: HF fast tokenizers/processors carry
    internal state, so a concurrency bug would surface as a mismatched or
    corrupted result here."""
    tokenizer, processor = loaded
    num_samples = 64

    bases = [
        _make_sample(
            tokenizer,
            processor,
            color=(i * 3 % 256, (i * 7) % 256, (i * 13) % 256),
            size=(64 + (i % 5) * 28, 64 + (i % 3) * 28),  # vary token/patch counts
            text=f"Question number {i}: what color dominates?",
        )
        for i in range(num_samples)
    ]

    # ground-truth references computed serially with the synchronous helper
    references = []
    for base in bases:
        s = copy.deepcopy(base)
        ids = sglang_rollout._prepare_prompt_ids(s, tokenizer, processor)
        references.append((ids, s.multimodal_train_inputs))

    async_samples = [copy.deepcopy(base) for base in bases]

    async def run_all():
        return await asyncio.gather(
            *(sglang_rollout._prepare_prompt_ids_async(s, tokenizer, processor) for s in async_samples)
        )

    async_ids = asyncio.run(run_all())

    for i, (sample, ids) in enumerate(zip(async_samples, async_ids, strict=True)):
        ref_ids, ref_mm = references[i]
        assert ids == ref_ids, f"sample {i}: prompt_ids mismatch under concurrency"
        _assert_deep_equal(ref_mm, sample.multimodal_train_inputs, f"sample[{i}].multimodal_train_inputs")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
