"""CPU tests for asynchronous multimodal rollout preprocessing.

These tests are hermetic (no GPU / real model required). They pin down the
*contract* that the async path must be a drop-in, semantics-preserving
replacement for the original synchronous path:

  * same ``prompt_ids`` and same ``sample.multimodal_train_inputs`` for every
    branch (fresh multimodal / reuse-existing-tokens / text-only);
  * ``asyncio.gather`` keeps ``image_data`` in submit order even when later
    images finish encoding first;
  * concurrent samples that share one processor never cross-contaminate;
  * the blocking work runs off the asyncio event-loop thread.

The real-processor equivalence (identical HF tensors) lives in
``test_async_multimodal_real_processor.py``; the payload-to-SGLang parity lives
in ``test_async_multimodal_request_parity.py``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from slime.rollout import sglang_rollout
from slime.utils import processing_utils
from slime.utils.types import Sample

NUM_GPUS = 0


class _FakeProcessor:
    """Deterministic stand-in for an HF processor.

    Mirrors the real output shape (``input_ids`` as a batched list, plus
    modality tensors) and records how many calls overlapped so tests can assert
    that work actually runs concurrently.
    """

    def __init__(self, work_seconds: float = 0.0) -> None:
        self.calls = 0
        self.max_concurrent = 0
        self._active = 0
        self._lock = threading.Lock()
        self._work_seconds = work_seconds

    def __call__(self, *, text, **kwargs):
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.calls += 1
        if self._work_seconds:
            time.sleep(self._work_seconds)  # releases the GIL -> real overlap
        try:
            num_images = len(kwargs.get("images", []) or [])
            return {
                "input_ids": [[ord(c) for c in text]],
                "attention_mask": [[1] * len(text)],
                "pixel_values": f"px::{text}",
                "image_grid_thw": f"thw::{num_images}",
            }
        finally:
            with self._lock:
                self._active -= 1


def _fail_tokenizer():
    return SimpleNamespace(encode=lambda *_a, **_k: pytest.fail("tokenizer should not be called"))


# ---------------------------------------------------------------------------
# Off-event-loop guarantees (kept from the original suite)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_async_image_encoding_matches_sync_and_runs_off_event_loop(monkeypatch):
    image = Image.new("RGB", (2, 2), color="red")
    expected = processing_utils.encode_image_for_rollout_engine(image)
    event_loop_thread = threading.get_ident()
    worker_threads = []
    original_encode = processing_utils.encode_image_for_rollout_engine

    def tracked_encode(value):
        worker_threads.append(threading.get_ident())
        return original_encode(value)

    monkeypatch.setattr(processing_utils, "encode_image_for_rollout_engine", tracked_encode)

    actual = asyncio.run(processing_utils.async_encode_image_for_rollout_engine(image))

    assert actual == expected
    assert worker_threads and worker_threads[0] != event_loop_thread


@pytest.mark.unit
def test_async_prompt_preparation_runs_processor_off_event_loop():
    event_loop_thread = threading.get_ident()
    processor_threads = []

    def processor(*, text, **kwargs):
        processor_threads.append(threading.get_ident())
        assert text == "prompt"
        assert kwargs["images"] == ["image"]
        return {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "pixel_values": "pixels",
        }

    sample = Sample(prompt="prompt", multimodal_inputs={"images": ["image"]})
    tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: pytest.fail("tokenizer should not be called"))

    prompt_ids = asyncio.run(sglang_rollout._prepare_prompt_ids_async(sample, tokenizer, processor))

    assert prompt_ids == [1, 2, 3]
    assert sample.multimodal_train_inputs == {"pixel_values": "pixels"}
    assert processor_threads and processor_threads[0] != event_loop_thread


@pytest.mark.unit
def test_async_prompt_preparation_preserves_text_only_fast_path():
    encode_calls = []
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: encode_calls.append((text, add_special_tokens)) or [4, 5]
    )

    def processor(**_kwargs):
        pytest.fail("processor should not be called")

    sample = Sample(prompt="text only")

    prompt_ids = asyncio.run(sglang_rollout._prepare_prompt_ids_async(sample, tokenizer, processor))

    assert prompt_ids == [4, 5]
    assert encode_calls == [("text only", False)]


# ---------------------------------------------------------------------------
# Semantic equivalence: sync _prepare_prompt_ids  ==  async _prepare_prompt_ids_async
# The sync helper still lives in the tree, so it is the ground-truth reference.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_async_matches_sync_on_fresh_multimodal_branch():
    proc_sync = _FakeProcessor()
    proc_async = _FakeProcessor()
    sample_sync = Sample(prompt="hello world", multimodal_inputs={"images": ["imgA", "imgB"]})
    sample_async = Sample(prompt="hello world", multimodal_inputs={"images": ["imgA", "imgB"]})

    ids_sync = sglang_rollout._prepare_prompt_ids(sample_sync, _fail_tokenizer(), proc_sync)
    ids_async = asyncio.run(sglang_rollout._prepare_prompt_ids_async(sample_async, _fail_tokenizer(), proc_async))

    assert ids_sync == ids_async
    # the processor-derived training inputs must be byte-for-byte identical
    assert sample_sync.multimodal_train_inputs == sample_async.multimodal_train_inputs
    assert sample_async.multimodal_train_inputs == {
        "pixel_values": "px::hello world",
        "image_grid_thw": "thw::2",
    }
    assert proc_sync.calls == proc_async.calls == 1


@pytest.mark.unit
def test_async_matches_sync_on_reuse_existing_tokens_branch():
    proc_sync = _FakeProcessor()
    proc_async = _FakeProcessor()
    kwargs = dict(
        prompt="x",
        tokens=[9, 9, 9],
        multimodal_inputs={"images": ["i"]},
        multimodal_train_inputs={"pixel_values": "precomputed"},
    )
    sample_sync = Sample(**kwargs)
    sample_async = Sample(**kwargs)

    ids_sync = sglang_rollout._prepare_prompt_ids(sample_sync, _fail_tokenizer(), proc_sync)
    ids_async = asyncio.run(sglang_rollout._prepare_prompt_ids_async(sample_async, _fail_tokenizer(), proc_async))

    assert ids_sync == ids_async == [9, 9, 9]
    # reuse path must not re-run the processor and must not clobber existing inputs
    assert proc_sync.calls == proc_async.calls == 0
    assert sample_async.multimodal_train_inputs == {"pixel_values": "precomputed"}


@pytest.mark.unit
def test_async_matches_sync_when_multimodal_train_inputs_already_set_but_no_tokens():
    # tokens empty -> cannot reuse -> processor re-runs for prompt_ids, but an
    # already-populated multimodal_train_inputs must be preserved, not overwritten.
    proc_sync = _FakeProcessor()
    proc_async = _FakeProcessor()
    kwargs = dict(
        prompt="abc",
        multimodal_inputs={"images": ["i"]},
        multimodal_train_inputs={"pixel_values": "keep-me"},
    )
    sample_sync = Sample(**kwargs)
    sample_async = Sample(**kwargs)

    ids_sync = sglang_rollout._prepare_prompt_ids(sample_sync, _fail_tokenizer(), proc_sync)
    ids_async = asyncio.run(sglang_rollout._prepare_prompt_ids_async(sample_async, _fail_tokenizer(), proc_async))

    assert ids_sync == ids_async == [ord(c) for c in "abc"]
    assert proc_sync.calls == proc_async.calls == 1
    assert sample_async.multimodal_train_inputs == {"pixel_values": "keep-me"}


# ---------------------------------------------------------------------------
# asyncio.gather keeps image_data in SUBMIT order, not completion order.
# This is what guarantees the request byte stream is unchanged for multi-image
# samples.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_gather_preserves_image_order_under_reversed_completion(monkeypatch):
    # r-channel encodes the image index * 10 so slow_encode can recover it
    images = [Image.new("RGB", (2, 2), color=(i * 10, 0, 0)) for i in range(5)]
    original_encode = processing_utils.encode_image_for_rollout_engine
    reference = [original_encode(im) for im in images]

    completion_order = []

    def slow_encode(image):
        idx = image.getpixel((0, 0))[0]
        time.sleep((60 - idx) / 500.0)  # earlier index -> slower -> finishes last
        result = original_encode(image)
        completion_order.append(idx)
        return result

    monkeypatch.setattr(processing_utils, "encode_image_for_rollout_engine", slow_encode)

    async def gather_encode():
        return await asyncio.gather(*(processing_utils.async_encode_image_for_rollout_engine(im) for im in images))

    result = asyncio.run(gather_encode())

    assert result == reference, "gather must return image_data in submit order"
    # sanity: completion really was out of submit order (reversed here)
    assert completion_order == sorted(completion_order, reverse=True)


# ---------------------------------------------------------------------------
# Concurrency correctness: many distinct samples sharing one processor must each
# get their own correct result (no shared-state races through the executor).
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_concurrent_prompt_preparation_has_no_cross_contamination():
    proc = _FakeProcessor(work_seconds=0.005)
    num_samples = 16
    samples = [Sample(prompt=f"prompt-{i:03d}", multimodal_inputs={"images": [f"im{i}"]}) for i in range(num_samples)]

    async def run_all():
        return await asyncio.gather(
            *(sglang_rollout._prepare_prompt_ids_async(s, _fail_tokenizer(), proc) for s in samples)
        )

    results = asyncio.run(run_all())

    for i, (sample, prompt_ids) in enumerate(zip(samples, results, strict=True)):
        expected_text = f"prompt-{i:03d}"
        assert prompt_ids == [ord(c) for c in expected_text]
        assert sample.multimodal_train_inputs == {
            "pixel_values": f"px::{expected_text}",
            "image_grid_thw": "thw::1",
        }
    assert proc.calls == num_samples
    assert proc.max_concurrent > 1, "processor calls should actually overlap on the executor"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
